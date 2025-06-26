#!/bin/bash

# Kiểm tra và tạo virtual environment nếu chưa tồn tại
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Cài đặt dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Kiểm tra và cài đặt VnCoreNLP nếu cần
if [ ! -f "vncorenlp/VnCoreNLP-1.1.1.jar" ]; then
    echo "Setting up VnCoreNLP..."
    bash setup.sh
fi

# Cài đặt serve nếu chưa có
if ! command -v serve &> /dev/null; then
    echo "Installing serve..."
    npm install -g serve
fi

# Chạy frontend server trong background
echo "Starting frontend server..."
cd frontend
serve -s . -l 8080 &
FRONTEND_PID=$!
cd ..

# Chạy backend server với port 5001
echo "Starting backend server..."
echo "Frontend available at: http://localhost:8080"
echo "Backend API available at: http://localhost:5001"
python backend/main.py --port 5001 --host 127.0.0.1

# Cleanup khi script kết thúc
trap "kill $FRONTEND_PID" EXIT