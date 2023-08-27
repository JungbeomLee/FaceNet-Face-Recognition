# FaceNet-Face-Recognition
> FaceNet을 사용해 Siamese nueral network로 만든 face recognition 기능을 향상시킨 프로젝트이다.


# 모델 다운로드
> Hiroki Taniai가 제공하는 모델을 사용하였다.
> https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn


# 사용 방법
> 'image_embeding.py'에 있는 'FaceEmbedder' Class를 사용한다.
```python
embeding = FaceEmbedder(model_path = '--your model path')
embeding.get_embedded_face("--your image path")
```

> 더욱 자세한 사용 방법은 'function_test_code.py' 참고
