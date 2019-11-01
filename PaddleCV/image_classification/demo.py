from classifier import Classifier

mymodel = Classifier(work_dir="myproject", model_name="ResNet18", use_pretrained_weights=True, num_classes=114)
print(mymodel.pretrained_weights_dir)
mymodel.fit(data_dir="/ssd3/jiangxiaobai/data0926", num_epochs=2, lr=0.05, batch_size=32)

print("===============eval=============================")
mymodel.eval(data_dir="/ssd3/jiangxiaobai/data0926")
print('________________________________infer__________________________')
mymodel.predict(img_file='/ssd3/jiangxiaobai/data0926/train/all1/train/0/100.jpg')
