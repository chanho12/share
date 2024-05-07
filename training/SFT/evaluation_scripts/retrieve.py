

VECTOR_SEARCH_TOP_K = 3
CHUNK_SIZE = 200


class LocalMemoryRetrieval:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL_CN,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn'
                 ):
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k
    
    def init_memory_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    user_name: str = None,
                                    cur_date: str = None):
        loaded_files = []
        # filepath = filepath.replace('user',user_name)
        # vs_path = vs_path.replace('user',user_name)
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                # try:
                docs = load_memory_file(filepath,user_name,self.language)
                print(f"{file} 已成功加载")
                loaded_files.append(filepath)
                # except Exception as e:
                #     print(e)
                #     print(f"{file} 未能成功加载")
                #     return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    # if user_name not in fullfilepath:
                    #     continue
                    try:
                        docs += load_memory_file(fullfilepath,user_name,self.language)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_memory_file(file,user_name,self.language)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f'Load from previous memory index {vs_path}.')
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files
    
    def load_memory_index(self,vs_path):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size=self.chunk_size
        return vector_store
    
    def search_memory(self,
                    query,
                    vector_store):

        # vector_store = FAISS.load_local(vs_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        # vector_store.chunk_size=self.chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        pre_date = ''
        date_docs = []
        dates = []
        for doc in related_docs:
            doc.page_content = doc.page_content.replace(f'时间{doc.metadata["source"]}的对话内容：','').strip()
            if doc.metadata["source"] != pre_date:
                # date_docs.append(f'在时间{doc.metadata["source"]}的回忆内容是：{doc.page_content}')
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}' 
        # memory_contents = [doc.page_content for doc in related_docs]
        # memory_contents = [f'在时间'+doc.metadata['source']+'的回忆内容是：'+doc.page_content for doc in related_docs]
        return date_docs, ', '.join(dates) 
    

