import argparse
import os
from models import Worker

parser = argparse.ArgumentParser(description='Record Analyzer')
parser.add_argument('audio_file', type=str, help='Input audio file')
args = parser.parse_args()
audio_file = args.audio_file
print('Обработка файла', audio_file)


print('Инициализация')
worker = Worker()
print('Предобработка и транскрибация записи')
metadata_file, transcription_file, channels_processed = worker.process_audio_file(audio_file)
print('Анализ тональности')
tonality_report, tonality_plot = worker.analyze_wav(channels_processed)
print('Анализ текста')
sentence_report = worker.analyze_text(transcription_file)
print('Генерация отчета')
zip_path = worker.create_report(metadata_file, transcription_file, tonality_report, sentence_report, tonality_plot)
print('Очистка временных файлов')
worker.clean_temp_files()
print('Создан отчет', zip_path)