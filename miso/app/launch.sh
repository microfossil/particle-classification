#!/bin/bash

tmux new-session -d -s flask
tmux send-keys -t flask "python app.py" Enter

tmux new-session -d -s gpu_worker
tmux send-keys -t gpu_worker "celery -A app.celery_app worker --loglevel INFO -P threads -Q celery_gpu -n gpu_worker@%h" Enter

tmux new-session -d -s flower
tmux send-keys -t flower "celery -A app.celery_app flower" Enter

