import os
import stat
import sys

__dir__ = os.path.dirname(__file__)

# print(sys.path)
# print(os.path.expanduser("~"))
# print(os.path.basename(__file__))
# print(os.path.dirname(__file__))
# print(os.path.abspath(__file__))
st = os.stat("/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/Bank of America.pdf")
print(stat.S_ISREG(st.st_mode))


BASE_DIR = os.path.expanduser("~/.paddleocr/")
# print(BASE_DIR)
