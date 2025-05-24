import sys
from pathlib import Path

# Получаем абсолютный путь к корню проекта
project_root = Path(__file__).parent  # your_project/
sys.path.append(str(project_root))    # Добавляем корень в PYTHONPATH

# Теперь можно импортировать модули от корня
from modules.modules_kidera.kidera import kidera_final_dict
#from code.code_kidera.preprocessing.preprocessing import some_function

print("Импорт успешен!")