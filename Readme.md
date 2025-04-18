# neuro_quality_control / нейро_контроль_качества

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Maintenance](https://img.shields.io/badge/Maintained--yes-brightgreen.svg)

## [🇬🇧 English](#english) | 🇷🇺 Русский

---

## 🇷🇺 О проекте

**neuro\_quality\_control** - это Python-скрипт, предназначенный для анализа аудиозаписей телефонных разговоров с целью контроля качества работы менеджеров колл-центра. Он выполняет следующие ключевые функции:

* **Предобработка аудио:** Разделение аудио на каналы, применение фильтров для улучшения качества звука.

* **Транскрибация аудио:** Автоматическое преобразование речи в текст с использованием модели WhisperX.

* **Анализ тональности:** Определение эмоциональной окраски речи менеджера и клиента на протяжении разговора с визуализацией результатов.

* **Анализ текста:** Оценка работы менеджера на основе текста транскрипции с использованием языковой модели (через LM Studio) и базы знаний, поиск релевантных критериев оценки.

* **Генерация отчета:** Создание подробного отчета в формате DOCX и архивация его вместе с графиком тональности в ZIP-файл.

* **Очистка временных файлов:** Автоматическое удаление созданных в процессе работы временных файлов.

Проект **neuro\_quality\_control** может быть полезен для автоматизированной оценки качества обслуживания клиентов в колл-центрах, выявления проблемных зон в работе менеджеров и предоставления им обратной связи для улучшения.

### Ключевые особенности:

* Использование современных библиотек для обработки аудио и текста (ffmpeg, librosa, whisperx, transformers, sentence-transformers, faiss, docx, zipfile, lmstudio).

* Модульная структура кода, облегчающая понимание и расширение функциональности.

* Автоматизированный процесс анализа от загрузки аудио до формирования готового отчета.

* Визуализация результатов анализа тональности.

* Интеграция с базой знаний для более точной оценки работы менеджеров.

### Запуск проекта:

1.  Убедитесь, что у вас установлен Python 3.8 или выше.

2.  Установите необходимые зависимости, выполнив команду:
    ```bash
    pip install -r requirements.txt
    ```
    
3.  Запустите скрипт `main.py`, передав в качестве аргумента путь к аудиофайлу:

    ```bash

    python main.py path/to/your/audio_file.mp3

    ```
4.  Для запуска из Jupyter Notebook используйте `main.ipynb`.

### Структура проекта:


* `call-center-knowledge-base/`: Директория с файлами векторной базы знаний.

* `temp/`: Временная директория для хранения промежуточных файлов.

* `main.ipynb`: Jupyter Notebook для запуска анализа.

* `main.py`: Основной скрипт для запуска анализа из командной строки.

* `models.py`: Файл, содержащий классы и методы для обработки аудио, анализа и генерации отчетов.

* `README.md`: Текущий файл с описанием проекта.

* `requirements.txt`: Файл с перечнем необходимых Python-библиотек.

### Демонстрация работы (для презентации):

1.  Запустите `main.ipynb`.

2.  Предоставьте путь к тестовому аудиофайлу (например, `r"./10.10.2024__12-20-20______12.mp3"`).

3.  Дождитесь завершения работы скрипта.

4.  В выводе будет предоставлена ссылка для скачивания ZIP-архива с отчетом (`report.zip`).

5.  Распакуйте архив и продемонстрируйте сгенерированный отчет в формате DOCX, содержащий транскрипцию, результаты анализа тональности (текстовое описание и график), а также оценку работы менеджера и рекомендации на основе анализа текста.

---

## <a name="english"></a>🇬🇧 About the Project

**neuro\_quality\_control** is a Python script designed to analyze audio recordings of phone conversations for the purpose of quality control of call center managers' work. It performs the following key functions:

* **Audio Preprocessing:** Separating audio into channels and applying filters to improve sound quality.

* **Audio Transcription:** Automatically converting speech to text using the WhisperX model.

* **Sentiment Analysis:** Determining the emotional tone of the manager and the client throughout the conversation, with visualization of the results.

* **Text Analysis:** Evaluating the manager's performance based on the transcription text using a language model (via LM Studio) and a knowledge base, searching for relevant evaluation 
criteria.

* **Report Generation:** Creating a detailed report in DOCX format and archiving it along with the sentiment analysis graph into a ZIP file.

* **Temporary File Cleanup:** Automatically deleting temporary files created during the process.

The **neuro\_quality\_control** project can be useful for automated evaluation of customer service quality in call centers, identifying problem areas in managers' work, and providing them with feedback for improvement.

### Key Features:

* Utilizes modern libraries for audio and text processing (ffmpeg, librosa, whisperx, transformers, sentence-transformers, faiss, docx, zipfile, lmstudio).

* Modular code structure, making it easy to understand and extend functionality.

* Automated analysis process from audio loading to generating a complete report.

* Visualization of sentiment analysis results.

* Integration with a knowledge base for more accurate evaluation of managers' performance.

### Running the Project:

1.  Make sure you have Python 3.8 or higher installed.

2.  Install the necessary dependencies by running the command:

    ```bash

    pip install -r requirements.txt

    ```
    
3.  Run the `main.py` script, passing the path to the audio file as an argument:

    ```bash

    python main.py path/to/your/audio_file.mp3

    ```
4.  To run from Jupyter Notebook, use `main.ipynb`.

### Project Structure:

* `call-center-knowledge-base/`: Directory containing the vector knowledge base files.

* `temp/`: Temporary directory for storing intermediate files.

* `main.ipynb`: Jupyter Notebook for running the analysis.

* `main.py`: Main script for running the analysis from the command line.

* `models.py`: File containing classes and methods for audio processing, analysis, and report generation.

* `README.md`: This file describing the project.

* `requirements.txt`: File listing the required Python libraries.

### Demonstration (for presentation):

1.  Run `main.ipynb`.

2.  Provide the path to a test audio file (e.g., `r"./10.10.2024__12-20-20______12.mp3"`).

3.  Wait for the script to finish processing.

4.  The output will provide a download link for a ZIP archive with the report (`report.zip`).

5.  Unzip the archive and demonstrate the generated DOCX report, which includes the transcription, sentiment analysis results (textual description and graph), as well as the evaluation of the manager's performance and recommendations based on the text analysis.
