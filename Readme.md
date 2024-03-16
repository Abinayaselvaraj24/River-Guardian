## Building Plastic Free Rivers with AI hackathon conducted by Reva University

--------------------------------------



## Guides to run our project:

### Project directory Structure

<pre>
├── Artifacts
     └──best.onnx
├── predict_data
├── predicted_images
├── Utility_dir
     └──models
     └──utils
├── README.md
├── predict_pipeline.py
├── get_description.py
├── create_csv.py
└── app.py

</pre>


### Create new virtual environment`
```
python -m venv env
```


### Install all dependencies
```
pip install -r requirements.txt
```



We made the prediction pipeline to run both single image and for image directory

```
python predict_pipeline.py --input_path path --output_dir path
```

### Example for directory
```
python predict_pipeline.py --input_path   predict_data  --output_dir  predicted_images
```

### Example for Single image

```
python predict_pipeline.py --input_path   predict_data/DJI_0185.jpg  --output_dir  predicted_images
```



