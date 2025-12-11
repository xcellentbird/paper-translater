import sys
from pathlib import Path

# 프로젝트 루트 디렉터리를 sys.path 에 추가하여
# `from utils.pdf_loader import ...` 와 같은 임포트가 가능하도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
