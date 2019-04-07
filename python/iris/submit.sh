gcloud.cmd ml-engine jobs submit training pytest000002 \
    --module-name iris.iris_train \
    --region us-central1 \
    --package-path iris \
    --python-version 3.5 \
    --job-dir gs://keras-235720/py-cloudml

# Run from inside the iris directory
gcloud.cmd ml-engine local train \
    --module-name iris.iris_train \
    --package-path iris \
    --job-dir output
