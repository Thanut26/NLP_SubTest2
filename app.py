import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Stock Investment AI", page_icon="✨", layout="centered")

# ฝัง CSS ตกแต่ง UI นำเข้าฟอนต์ Prompt และตั้งค่าธีมสีฟ้า
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .gradient-text {
        text-align: center;
        background: linear-gradient(90deg, #0052D4 0%, #4364F7 50%, #6FB1FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 600;
        margin-bottom: 0px;
    }
    .sub-text {
        text-align: center;
        color: #666666;
        font-size: 1.05em;
        margin-bottom: 2em;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# โหลด Groq API Key
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("ไม่พบ GROQ_API_KEY ใน Streamlit Secrets! กรุณาตั้งค่าก่อนใช้งาน")
    st.stop()

client = Groq(api_key=groq_api_key)

# โหลด Embedding Model และเตรียม Vector Database
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

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("✨ เมนูการใช้งาน")
    
    # ปุ่มแชทใหม่
    if st.button("➕ แชทใหม่ (New Chat)", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    
    # 1. ส่วนแสดงประวัติการแชท (จำลองแบบ Gemini)
    st.markdown("### 🕒 คำถามล่าสุด")
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        # ดึงมาเฉพาะคำถามของผู้ใช้ (ย้อนหลัง 5 ข้อล่าสุด)
        user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        for q in reversed(user_questions[-5:]):
            # ตัดความยาวคำถามไม่ให้ล้นบรรทัด
            short_q = q if len(q) < 22 else q[:22] + "..."
            st.markdown(f"<div style='font-size: 0.9em; color: #555; margin-bottom: 8px; padding: 8px; border-radius: 8px; background-color: #f8f9fa;'>💬 {short_q}</div>", unsafe_allow_html=True)
    else:
        st.caption("ยังไม่มีการสนทนา")
        
    st.markdown("---")
    
    # 2. ซ่อนรายชื่อไฟล์ไว้ในกล่องพับได้ (Expander)
    with st.expander("📚 ดูฐานข้อมูลของบอต", expanded=False):
        if doc_sources:
            unique_sources = sorted(list(set(doc_sources)))
            files_html = ""
            for src in unique_sources:
                files_html += f"""
                <div style='padding: 6px 10px; margin-bottom: 6px; border-radius: 6px; 
                            background-color: #ffffff; border: 1px solid #E4E7EB; 
                            font-size: 0.85em; color: #4A5568; display: flex; align-items: center;'>
                    <span style='margin-right: 8px; font-size: 1.1em;'>📄</span> {src}
                </div>
                """
            st.markdown(files_html, unsafe_allow_html=True)
        else:
            st.warning("ยังไม่มีไฟล์ข้อมูล")
            
    st.markdown("---")
    
    # ปุ่มอัปเดตข้อมูล
    st.caption("หากเพิ่มไฟล์ .txt ใหม่ ให้กดปุ่มนี้เพื่ออัปเดต")
    if st.button("🔄 รีโหลดฐานข้อมูล", use_container_width=True):
        load_rag_system.clear()
        st.rerun()
# ----------------------------------------------------
        
    else:
        st.warning("ยังไม่มีไฟล์ข้อมูล")
        
    st.markdown("---")
    
    # ปุ่มอัปเดตข้อมูลแบบไม่ต้องปิดเซิร์ฟเวอร์
    st.markdown("### 🔄 อัปเดตข้อมูล")
    st.caption("หากเพิ่มไฟล์ .txt ใหม่ ให้กดปุ่มนี้เพื่อให้บอตเรียนรู้ข้อมูลใหม่ทันที")
    if st.button("รีโหลดฐานข้อมูล", use_container_width=True):
        load_rag_system.clear()
        st.rerun()
# ----------------------------------------------------

if index is None or len(docs) == 0:
    st.warning("⚠️ ไม่พบไฟล์เอกสารในโฟลเดอร์ data/ กรุณาตรวจสอบอีกครั้ง")
    st.stop()

st.markdown('<p class="gradient-text">✨ Stock Investment AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">ผู้ช่วยอัจฉริยะเรื่องหุ้นและการลงทุน พร้อมตอบทุกข้อสงสัยของคุณ</p>', unsafe_allow_html=True)

# เก็บประวัติการแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการแชทเดิม
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# รับข้อความจากผู้ใช้
if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับการลงทุนในหุ้นที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและประมวลผลคำตอบ..."):
            question_embedding = model.encode([prompt], convert_to_numpy=True)
            
            k = min(3, len(docs))
            distances, indices = index.search(question_embedding, k)
            
            retrieved_context = ""
            retrieved_sources = set()
            
            for i, idx in enumerate(indices[0]):
                if distances[0][i] < 2.5: 
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
                    "หากคำตอบอยู่ใน Context ให้สรุปตอบให้ชัดเจน อ่านง่าย เป็นธรรมชาติ "
                    "และห้ามแต่งข้อมูลที่นอกเหนือจาก Context หากไม่มีคำตอบให้ตอบว่า 'ไม่พบข้อมูล'"
                )
                user_prompt = f"Context:\n{retrieved_context}\n\nQuestion: {prompt}"

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="openai/gpt-oss-20b",
                temperature=0.1,
            )
            
            answer = chat_completion.choices[0].message.content

            if retrieved_sources and "ไม่พบข้อมูล" not in answer:
                sources_str = ", ".join(retrieved_sources)
                full_response = f"{answer}\n\n*📌 แหล่งอ้างอิง: {sources_str}*"
            else:
                full_response = answer

            st.markdown(full_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})