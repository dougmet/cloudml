# Submit from inside the iris directory (where this sh file is)

# Submit to ML Engine
gcloud ml-engine jobs submit training pytest000002 \
    --module-name iris.iris_train \
    --region us-central1 \
    --package-path iris \
    --python-version 3.5 \
    --job-dir gs://keras-235720/py-cloudml

# Local Run
gcloud ml-engine local train \
    --module-name iris.iris_train \
    --package-path iris \
    --job-dir output
