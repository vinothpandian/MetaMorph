import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="localhost", port=8000, reload=True)
