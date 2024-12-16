from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from utils.sootiai_web import Agent

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app)

agent = Agent(base_url="http://localhost:5000/v1", api_key="OPEN_API_KEY")
stop_task_flag = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/execute_task', methods=['POST'])
def execute_task():
    task = request.json.get('task')
    try:
        agent.execute_task(task, stop_task_flag)
        return jsonify({'status': 'success', 'message': 'Task executed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/clear', methods=['POST'])
def clear():
    try:
        agent.global_history = []  # Clear global history
        return jsonify({'status': 'success', 'message': 'Tasks cleared successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@socketio.on('send_message')
def handle_send_message(data):
    task = data.get('task')
    try:
        # Execute the task
        agent.execute_task(task, stop_task_flag)
        emit('receive_message', {'status': 'completed', 'message': 'Task executed successfully'})
    except Exception as e:
        # Handle any errors during task execution
        emit('receive_message', {'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)