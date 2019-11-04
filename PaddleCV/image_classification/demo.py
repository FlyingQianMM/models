from image_classification.classifier import Classifier

# The fisrt using method
# mymodel = Classifier(work_dir="myproject", model_name="ResNet18", use_pretrained_weights=True, num_classes=114)
# print(mymodel.pretrained_weights_dir)
# mymodel.fit(data_dir="/ssd3/jiangxiaobai/data0926", num_epochs=2, lr=0.05, batch_size=32)
# print("===============eval=============================")
# mymodel.eval(data_dir="/ssd3/jiangxiaobai/data0926")
# print('===============infer=============================')
# mymodel.predict(img_file='/ssd3/jiangxiaobai/data0926/train/all1/train/0/100.jpg')

# The second using method
model_path = '/all/image_classification/output/ResNet18/'
mymodel = Classifier(
    work_dir="myproject", model_name="ResNet18", num_classes=283)
mymodel.load_model(model_path)
print("===============eval=============================")
mymodel.eval(data_dir="/all/mini_data/")
print('===============infer=============================')
mymodel.predict('/all/mini_data/kmeans_data/0/9.jpg')
