class Prompts:
    """Storage for all LLM prompts in Uzbek language."""
    
    SEARCH_PROMPT = """Siz O'zbek tili bo'yicha yordamchi mutaxassisiz. 
Berilgan ma'lumotlar (kontekst) asosida foydalanuvchining savoliga aniq va to'liq javob bering.

Kontekst:
{context}

Savol: {question}

Javob:"""

    SUMMARY_PROMPT = """Quyidagi matnni O'zbek tilida qisqacha mazmunini yoritib bering:

Matn:
{text}

Xulosa:"""

    ROUTER_PROMPT = """Foydalanuvchi so'rovini tahlil qiling va quyidagi vositalardan birini tanlang:
1. 'SearchTool' - Agar so'rov bilimlar bazasidan ma'lumot qidirishga oid bo'lsa.
2. 'SummaryTool' - Agar so'rov matnni umumlashtirish yoki qisqartirishga oid bo'lsa.
3. 'SentimentTool' - Agar so'rov matnning kayfiyatini aniqlashga oid bo'lsa.

So'rov: {user_input}

Faqat vosita nomini qaytaring."""

prompts = Prompts()