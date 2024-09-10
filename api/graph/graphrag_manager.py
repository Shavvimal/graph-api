import asyncio
import os
import pandas as pd
import tiktoken
from dotenv import load_dotenv

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)

# parquet files generated from indexing pipeline
from api.utils.constants import (
    COMMUNITY_REPORT_TABLE,
    ENTITY_TABLE,
    ENTITY_EMBEDDING_TABLE,
    RELATIONSHIP_TABLE,
    COVARIATE_TABLE,
    TEXT_UNIT_TABLE,
)


# Load environment variables
_ = load_dotenv()


class GraphRAGManager:
    def __init__(self, root_path = "../../data", folder = "20240903-181408", community_level = 2):
        self._FOLDER = folder
        self._ROOT_PATH = root_path
        # community level in the Leiden community hierarchy from which we will load the community reports
        # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
        self._COMMUNITY_LEVEL = community_level
        self._INPUT_DIR = f"{self._ROOT_PATH}/{self._FOLDER}/artifacts"
        self._LANCEDB_URI = f"{self._INPUT_DIR}/lancedb"
        # Setup
        self.entity_df, self.entity_embedding_df, self.report_df, self.relationship_df, self.covariate_df, self.text_unit_df = self._load_parquet_files()
        self.llm, self.token_encoder, self.text_embedder = self._setup_llm()
        self.description_embedding_store = None

    def _setup_llm(self):
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        api_base = os.environ["GRAPHRAG_API_BASE"]
        api_version = os.environ["GRAPHRAG_API_VERSION"]
        llm_model = os.environ["GRAPHRAG_LLM_MODEL"]
        llm_deployment_name = os.environ["GRAPHRAG_LLM_DEPLOYMENT_NAME"]
        embedding_model = os.environ["GRAPHRAG_EMBEDDING_MODEL"]
        embedding_deployment_name = os.environ["GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME"]
        embedding_version = os.environ["GRAPHRAG_EMBEDDING_VERSION"]

        llm = ChatOpenAI(
            api_key=api_key,
            model=llm_model,
            api_type=OpenaiApiType.AzureOpenAI,
            deployment_name=llm_deployment_name,
            max_retries=20,
            api_base=api_base,
            api_version=api_version,
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")
        text_embedder = OpenAIEmbedding(
            api_key=api_key,
            api_base=api_base,
            api_type=OpenaiApiType.AzureOpenAI,
            model=embedding_model,
            deployment_name=embedding_deployment_name,
            max_retries=20,
            api_version=embedding_version,
        )

        return llm, token_encoder, text_embedder

    def _load_parquet_files(self, claim_extraction_enabled=True):
        entity_df = pd.read_parquet(f"{self._INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{self._INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        report_df = pd.read_parquet(f"{self._INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        relationship_df = pd.read_parquet(f"{self._INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        covariate_df = pd.read_parquet(f"{self._INPUT_DIR}/{COVARIATE_TABLE}.parquet") if claim_extraction_enabled else pd.DataFrame()
        text_unit_df = pd.read_parquet(f"{self._INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")

        return entity_df, entity_embedding_df, report_df, relationship_df, covariate_df, text_unit_df

    def _setup_graph_index(self):
        ################################################################
        ###################### Read entities ########################
        ################################################################

        entities = read_indexer_entities(self.entity_df, self.entity_embedding_df, self._COMMUNITY_LEVEL)

        # load description embeddings to an in-memory lancedb vectorstore
        # to connect to a remote db, specify url and port values.
        self.description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        self.description_embedding_store.connect(db_uri=self._LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=self.description_embedding_store)

        ################################################################
        ###################### Read relationships ######################
        ################################################################

        relationships = read_indexer_relationships(self.relationship_df)

        ################################################################
        ###################### Covariates ######################
        ################################################################

        # NOTE: covariates are turned off by default, because they generally need prompt tuning to be valuable
        # Please see the GRAPHRAG_CLAIM_* settings
        covariate_df = pd.read_parquet(f"{self._INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        covariates = {"claims": claims}

        ################################################################
        ###################### Read community reports ##################
        ################################################################

        report_df = pd.read_parquet(f"{self._INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, self.entity_df, self._COMMUNITY_LEVEL)

        #######################################################################
        ###################### Read text units ##################
        ################################################################

        text_unit_df = pd.read_parquet(f"{self._INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)

        return reports, text_units, entities, relationships, covariates

    def _create_local_context(self):
        ################################################################
        ################ Create local search context builder ###########
        ################################################################
        reports, text_units, entities, relationships, covariates = self._setup_graph_index()

        context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            # if you did not run covariates during indexing, set this to None
            covariates=covariates,
            # covariates=None,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )

        ####################################################################
        ##################### Create local search engine ###################
        ####################################################################

        # text_unit_prop: proportion of context window dedicated to related text units
        # community_prop: proportion of context window dedicated to community reports.
        # The remaining proportion is dedicated to entities and relationships. Sum of text_unit_prop and community_prop should be <= 1
        # conversation_history_max_turns: maximum number of turns to include in the conversation history.
        # conversation_history_user_turns_only: if True, only include user queries in the conversation history.
        # top_k_mapped_entities: number of related entities to retrieve from the entity description embedding store.
        # top_k_relationships: control the number of out-of-network relationships to pull into the context window.
        # include_entity_rank: if True, include the entity rank in the entity table in the context window. Default entity rank = node degree.
        # include_relationship_weight: if True, include the relationship weight in the context window.
        # include_community_rank: if True, include the community rank in the context window.
        # return_candidate_context: if True, return a set of dataframes containing all candidate entity/relationship/covariate records that
        # could be relevant. Note that not all of these records will be included in the context window. The "in_context" column in these
        # dataframes indicates whether the record is included in the context window.
        # max_tokens: maximum number of tokens to use for the context window.

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
            "max_tokens": 12_000,
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        }

        llm_params = {
            "max_tokens": 2_000,
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
            "temperature": 0.0,
        }

        return context_builder, llm_params, local_context_params

    def create_local_search_engine(self):
        """
        Local search method generates answers by combining relevant data from the AI-extracted knowledge-graph with text chunks of the raw documents. This method is suitable for questions that require an understanding of specific entities mentioned in the documents (e.g. What are the healing properties of chamomile?).
        """

        context_builder, llm_params, local_context_params = self._create_local_context()

        return LocalSearch(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
        )

    def create_question_generation_engine(self):
        """
        This function takes a list of user queries and generates the next candidate questions.
        """
        context_builder, llm_params, local_context_params = self._create_local_context()

        return LocalQuestionGen(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

    def _create_global_context(self):
        """
        - Load all community reports in the `create_final_community_reports` table from the ire-indexing engine, to be used as context data for global search.
        - Load entities from the `create_final_nodes` and `create_final_entities` tables from the ire-indexing engine, to be used for calculating community weights for context ranking. Note that this is optional (if no entities are provided, we will not calculate community weights and only use the `rank` attribute in the community reports table for context ranking)
        """

        reports, text_units, entities, relationships, covariates = self._setup_graph_index()

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities, # default to None if you don't want to use community weights for ranking
            token_encoder=self.token_encoder,
        )

        ####################################################################
        ##################### Create global search engine ###################
        ####################################################################

        context_builder_params = {
            "use_community_summary": False,
            # False means using full community reports. True means using community short summaries.
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12_000,
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 2000,
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
            "temperature": 0.0,
        }

        return context_builder, map_llm_params, reduce_llm_params, context_builder_params

    def create_global_search_engine(self):
        """
        Global search method generates answers by searching over all AI-generated community reports in a map-reduce fashion. This is a resource-intensive method, but often gives good responses for questions that require an understanding of the dataset as a whole (e.g. What are the most significant values of the herbs mentioned in this notebook?).
        """

        context_builder, map_llm_params, reduce_llm_params, context_builder_params = self._create_global_context()

        return GlobalSearch(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

if __name__ == '__main__':
    async def local_search_example():
        # Run local search on sample queries
        graphrag_manager = GraphRAGManager(folder="20240903-194043")
        local_search_engine = graphrag_manager.create_local_search_engine()
        result = await local_search_engine.asearch("Tell me about Antler")

        # Inspecting the context data used to generate the response
        # result.context_data["entities"].head()
        # result.context_data["relationships"].head()
        # result.context_data["reports"].head()
        # result.context_data["sources"].head()
        # if "claims" in result.context_data:
        #     print(result.context_data["claims"].head())
        #

        return result

    async def generate_questions():
        # Run question generation on sample queries
        graphrag_manager = GraphRAGManager()
        question_gen_engine = graphrag_manager.create_question_generation_engine()
        question_history = [
            "Tell me about Antler",
            "What is the best approach for early stage startups to raise capital?",
        ]
        candidate_questions = await question_gen_engine.agenerate(
            question_history=question_history, context_data=None, question_count=5
        )
        print(candidate_questions.response)

    async def global_search_example():
        # Run global search on sample queries
        graphrag_manager = GraphRAGManager()
        global_search_engine = graphrag_manager.create_global_search_engine()
        result = await global_search_engine.asearch(
            "What is the major lessons for Entrepreneurs?"
        )
        # print(result.response)
        # inspect the data used to build the context for the LLM responses
        # result.context_data["reports"]
        # inspect number of LLM calls and tokens
        # print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
        return result

    result = asyncio.run(local_search_example())
    print(result)

