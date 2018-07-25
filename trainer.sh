export FLAGS_fraction_of_gpu_memory_to_use=0.0
export FLAGS_rpc_deadline=6000000
export PADDLE_TRAINING_ROLE=TRAINER
export PADDLE_PSERVER_ENDPOINTS=127.0.0.1:5002,127.0.0.1:5003
export PADDLE_TRAINERS=2
export GLOG_logtostderr=1
export GLOG_v=3

PADDLE_CURRENT_ENDPOINT=127.0.0.1:5002 PADDLE_TRAINER_ID=0 CUDA_VISIBLE_DEVICES=0,1 python train.py &> t0 &
PADDLE_CURRENT_ENDPOINT=127.0.0.1:5003 PADDLE_TRAINER_ID=1 CUDA_VISIBLE_DEVICES=2,3 python train.py &> t1 &
