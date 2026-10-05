# Stock Investment RAG Chatbot

Web Application ระบบ RAG (Retrieval-Augmented Generation) สำหรับให้ความรู้และตอบคำถามพื้นฐานเกี่ยวกับการลงทุนในหุ้นและตลาดหลักทรัพย์

## หัวข้อที่เลือก
- ผู้ช่วยให้ความรู้เรื่องหุ้นและการลงทุน (Stock Investment Knowledge Assistant)

## เทคโนโลยีที่ใช้
- Streamlit (สำหรับสร้าง Web Interface)
- Sentence-Transformers (`all-MiniLM-L6-v2`) สำหรับแปลง Embedding
- FAISS สำหรับทำ Vector Search ค้นหาข้อมูล
- Groq API (`llama-3.1-8b-instant`) สำหรับประมวลผลและสร้างคำตอบ (LLM)

## วิธีการใช้งาน
1. พิมพ์คำถามเกี่ยวกับหุ้นหรือตลาดหลักทรัพย์ลงในช่องแชท
2. ระบบจะทำการดึงข้อมูลจากเอกสารความรู้ในโฟลเดอร์ `data/` มาวิเคราะห์และตอบกลับพร้อมแหล่งอ้างอิง
3. หากไม่มีข้อมูลในระบบ ระบบจะตอบว่า "ไม่พบข้อมูล" ตามเงื่อนไข RAG