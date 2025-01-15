from sentence_transformers import SentenceTransformer, util
from ollama import ChatResponse, chat
from qdrant_client import QdrantClient
import importlib
import torch

def deletarColecao(client, collectionName):
    try:
        client.delete_collection(collectionName=collectionName)
        print(f"Coleção: '{collectionName}' deletada com sucesso!")
    except Exception as e:
        print(f"Erro ao deletar a coleção '{collectionName}': {e}")


def armazenarEmbeddings():
    faq_embeddings = []
    for item in faq_list:
        pergunta_embedding = model.encode(item["pergunta"], convert_to_tensor=True).cpu()
        faq_embeddings.append({
            "pergunta": item["pergunta"],
            "resposta": item["resposta"],
            "pergunta_embedding": pergunta_embedding
        })

    collectionName = "faq_list"

    try:
        client.create_collection(
            collection_name=collectionName,
            vectors_config={"size": 384, "distance": "Cosine"}
        )
        print(f"Coleção '{collectionName}' criada com sucesso.")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Coleção '{collectionName}' já existe.")
        else:
            print(f"Erro ao criar coleção '{collectionName}': {e}")
            return

    for i, item in enumerate(faq_embeddings):
        try:
            client.upsert(
                collection_name=collectionName,
                points=[
                    {"id": i, "vector": item["pergunta_embedding"].tolist(), "payload": faq_list[i]}
                ]
            )
        except Exception as e:
            print(f"Erro ao inserir o ponto {i} na coleção '{collectionName}': {e}")

    print("Embeddings armazenados com sucesso no Qdrant!")



def buscarDocumentosRelevantes(query, top_k=3, similaridadeMinima=0.7):
    queryEmbedding = model.encode(query, convert_to_tensor=True).to(device)
    results = client.search(
        collection_name="faq_list",
        query_vector=queryEmbedding.tolist(),
        limit=top_k
    )
    print("Resultados encontrados:", results)  # Log para verificar resultados
    documentosRelevantes = [result.payload for result in results if result.score >= similaridadeMinima]

    return documentosRelevantes


def consultarModeloLocal(consultaUsuario, documentosRelevantes, historicoMensagens):
    if not documentosRelevantes:
        return "Desculpe, não encontrei uma resposta no FAQ para a sua pergunta. Tente reformular ou consultar o site oficial."

    dadosFAQ = "\n".join([doc['resposta'] for doc in documentosRelevantes])
    fullPrompt = (
        f"Dados do FAQ:\n{dadosFAQ}\n"
        f"Pergunta do usuário: {consultaUsuario}\n"
        "Atenção: Responda com base apenas nos dados fornecidos do FAQ. Não invente informações."
    )

    historicoMensagens.append({"role": "user", "content": consultaUsuario})

    response: ChatResponse = chat(model="llama3.2", messages=historicoMensagens + [
        {"role": "system", "content": "Você é um assistente útil que responde perguntas com base em FAQs fornecidas."},
        {"role": "user", "content": fullPrompt}
    ], options={"temperature": 0.6})

    respostaContent = response.message.content

    historicoMensagens.append({"role": "assistant", "content": respostaContent})

    return respostaContent


def executarFluxo(consultaUsuario, top_k=3, historicoMensagens=[]):
    documentosRelevantes = buscarDocumentosRelevantes(consultaUsuario, top_k)
    respostaFinal = consultarModeloLocal(consultaUsuario, documentosRelevantes, historicoMensagens)
    return respostaFinal


def inicializar():
    armazenarEmbeddings()


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient("http://localhost:6333")
faq_module = importlib.import_module("faq_gov")
faq_list = faq_module.faqGov
historicoMensagens = []

inicializar()

print("Bem-vindo ao sistema de FAQ. Faça sua pergunta ou digite 'q' para sair.")

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() == "q":
        print("Encerrando o sistema de FAQ. Até logo!")
        break

    respostaFinal = executarFluxo(pergunta, historicoMensagens=historicoMensagens)

    print(f"GOV: {respostaFinal}")
    
