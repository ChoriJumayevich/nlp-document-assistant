import os
from loguru import logger
from capstone.modules.vector_store import vector_store_manager
from capstone.modules.m15_langchain_agent import assistant_agent
from capstone.modules.config import settings

def initialize_app():
    """Tizimni ishga tushirish va indeksni tekshirish."""
    logger.info(f"{settings.PROJECT_NAME} ishga tushmoqda...")
    
    # Vektor bazasini yuklash yoki yaratish
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        logger.info("Vektor bazasi topilmadi. Indekslash boshlanmoqda...")
        vector_store_manager.index_kb()
    else:
        vector_store_manager.load_index()
    
    logger.info("Tizim foydalanishga tayyor.")

def main():
    """Asosiy CLI interfeysi."""
    initialize_app()
    
    print("\n" + "="*50)
    print(f"Xush kelibsiz! {settings.PROJECT_NAME}")
    print("Savolingizni yozing (chiqish uchun 'exit' deb yozing).")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("Siz: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'chiqish']:
                print("\nXayr! Salomat bo'ling.")
                break
                
            if not user_input:
                continue

            # Agentni ishga tushirish
            response = assistant_agent.run(user_input)
            
            print(f"\nYordamchi: {response}")
            
            # Tracingni ko'rsatish (ixtiyoriy, tahlil uchun)
            # print(f"\nTrace: {assistant_agent.last_trace()}")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nDastur to'xtatildi.")
            break
        except Exception as e:
            logger.error(f"Xatolik yuz berdi: {e}")
            print("Kechirasiz, kutilmagan xatolik yuz berdi.")

if __name__ == "__main__":
    main()