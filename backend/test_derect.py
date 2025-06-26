#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.api import TextSimilarityAPI

print("🧪 Testing Flask app directly...")

if __name__ == '__main__':
    try:
        # Tạo API instance trực tiếp
        api = TextSimilarityAPI()
        print("✅ API instance created successfully")
        
        # Chạy với các cài đặt khác nhau
        print("🚀 Starting Flask server...")
        api.app.run(
            host='127.0.0.1',  # Thử localhost trước
            port=5001,         # Đổi port
            debug=True,
            use_reloader=False,
            threaded=True
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()