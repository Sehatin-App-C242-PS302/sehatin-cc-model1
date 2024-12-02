# Gunakan Python runtime
FROM python:3.9-slim

# Direktori kerja
WORKDIR /app

# Salin semua file
COPY . /app

# Instal dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# Ekspor port
EXPOSE 8080

# Jalankan aplikasi
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
