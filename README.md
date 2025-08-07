1. Create a new virtual environment and install all the required packages in it (pip install -r requirements.txt)

2. Run the docker image in port 6333 and 6334, pull the latest image using command (docker pull qdrant/qdrant:latest)

3. Run the main file (python main.py --mode pipeline) to setup

4. To see the frontend results run (python main.py --mode streamlit)

5. To see the evaluation results/ type your own query in ther terminal run (python main.py --mode search)
