import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Stock Investment RAG Chatbot", page_icon="📈", layout="centered")

st.title("📈 ผู้ช่วยให้ความรู้เรื่องหุ้นและการลงทุน (Stock Investment RAG)")
st.write("แชตบอตอัจฉริยะที่ช่วยตอบคำถามพื้นฐานเกี่ยวกับการลงทุนในตลาดหุ้น")

# โหลด Groq API Key จาก st.secrets
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("ไม่พบ GROQ_API_KEY ใน Streamlit Secrets! กรุณาตั้งค่าก่อนใช้งาน")
    st.stop()

client = Groq(api_key=groq_api_key)

# โหลด Embedding Model และเตรียม Vector Database (ใช้ @st.cache_resource เพื่อโหลดครั้งเดียว)
@st.cache_resource
def load_rag_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    docs = []
    doc_sources = []
    
    data_dir = "data"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        docs.append(content)
                        doc_sources.append(filename)
                        
    if not docs:
        return model, None, [], []
        
    embeddings = model.encode(docs, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return model, index, docs, doc_sources

model, index, docs, doc_sources = load_rag_system()

if index is None or len(docs) == 0:
    st.warning("⚠️ ไม่พบไฟล์เอกสารในโฟลเดอร์ data/ กรุณาตรวจสอบอีกครั้ง")
    st.stop()

# เก็บประวัติการแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# รับข้อความจากผู้ใช้
if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับการลงทุนในหุ้นที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและประมวลผลคำตอบ..."):
            question_embedding = model.encode([prompt], convert_to_numpy=True)
            
            k = min(2, len(docs))
            distances, indices = index.search(question_embedding, k)
            
            retrieved_context = ""
            retrieved_sources = set()
            
            for i, idx in enumerate(indices[0]):
                if distances[0][i] < 1.2:
                    retrieved_context += f"- {docs[idx]}\n"
                    retrieved_sources.add(doc_sources[idx])
            
            if not retrieved_context.strip():
                system_prompt = (
                    "คุณคือผู้ช่วยอัจฉริยะ ตอบคำถามจาก Context ที่กำหนดให้เท่านั้น "
                    "หากใน Context ไม่มีคำตอบ ให้ตอบว่า 'ไม่พบข้อมูล' โดยห้ามเดาหรือแต่งคำตอบขึ้นมาเองเด็ดขาด"
                )
                user_prompt = f"Context:\n(ไม่มีข้อมูลที่เกี่ยวข้อง)\n\nQuestion: {prompt}"
            else:
                system_prompt = (
                    "คุณคือผู้ช่วยอัจฉริยะ ตอบคำถามโดยอิงจาก Context ที่กำหนดให้เท่านั้น "
                    "หากคำตอบอยู่ใน Context ให้สรุปตอบให้ชัดเจน และห้ามแต่งข้อมูลที่นอกเหนือจาก Context "
                    "หากไม่มีคำตอบใน Context ให้ตอบว่า 'ไม่พบข้อมูล'"
                )
                user_prompt = f"Context:\n{retrieved_context}\n\nQuestion: {prompt}"

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
            )
            
            answer = chat_completion.choices[0].message.content

            if retrieved_sources and "ไม่พบข้อมูล" not in answer:
                sources_str = ", ".join(retrieved_sources)
                full_response = f"{answer}\n\n📌 **แหล่งอ้างอิง:** `{sources_str}`"
            else:
                full_response = answer

            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})