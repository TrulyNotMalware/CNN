import tensorflow as tf
import tensorflow.keras.layers as layers

input = layers.Input(shape=(512,512,3))
#1층 시작 9x9 strides=2
t= layers.Conv2D(filters=64,kernel_size=(9,9),strides=2)(input)
t= layers.BatchNormalization(axis=3)(t)
t= layers.Activation('relu')(t)
t= layers.ZeroPadding2D(padding=(3))(t)
t= layers.MaxPooling2D(pool_size=3,strides=1)(t)

#1층 3갈래 커널 사이즈 9,7,5로 각각 큰거 작은거 중간거
first_1 = layers.Conv2D(filters=64,kernel_size=(7,7),strides=1,padding='SAME')(t)
first_1 = layers.BatchNormalization(axis=3)(first_1)

first_2 = layers.Conv2D(filters=64,kernel_size=(5,5),strides=1,padding='SAME')(t)
first_2 = layers.BatchNormalization(axis=3)(first_2)

first_3 = layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='SAME')(t)
first_3 = layers.BatchNormalization(axis=3)(first_3)
#1층 Add
t = layers.Add()([first_1,first_2,first_3,t])
t = layers.Activation('relu')(t)
t = layers.Dropout(0.3)(t)
#2층 시작, 7x7 strides=2
t = layers.Conv2D(filters=128,kernel_size=(7,7),strides=2)(t)
t = layers.BatchNormalization(axis=3)(t)
t = layers.Activation('relu')(t)
t = layers.ZeroPadding2D(padding=2)(t)
t = layers.MaxPooling2D(pool_size=2,strides=1)(t)
#2층 3갈래
second_1 = layers.Conv2D(filters=128,kernel_size=(7,7),strides=1,padding='SAME')(t)
second_1 = layers.BatchNormalization(axis=3)(second_1)

second_2 = layers.Conv2D(filters=128,kernel_size=(5,5),strides=1,padding='SAME')(t)
second_2 = layers.BatchNormalization(axis=3)(second_2)

second_3 = layers.Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='SAME')(t)
second_3 = layers.BatchNormalization(axis=3)(second_3)
#2층 Add
t = layers.Add()([second_1,second_2,second_3,t])
t = layers.Activation('relu')(t)
t = layers.Dropout(0.3)(t)
#3층 시작 5x5 strides=2
t = layers.Conv2D(filters=256,kernel_size=(5,5),strides=2)(t)
t = layers.BatchNormalization(axis=3)(t)
t = layers.Activation('relu')(t)
t = layers.ZeroPadding2D(padding=2)(t)
t = layers.MaxPooling2D(pool_size=2,strides=1)(t)
#3층 3갈래
third_1 = layers.Conv2D(filters=256,kernel_size=(5,5),strides=1,padding='SAME')(t)
third_1 = layers.BatchNormalization(axis=3)(third_1)

third_2 = layers.Conv2D(filters=256,kernel_size=(4,4),strides=1,padding='SAME')(t)
third_2 = layers.BatchNormalization(axis=3)(third_2)

third_3 = layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='SAME')(t)
third_3 = layers.BatchNormalization(axis=3)(third_3)
#3층 Add
t = layers.Add()([third_1,third_2,third_3,t])
t = layers.Activation('relu')(t)
t = layers.Dropout(0.3)(t)

#4층 시작, 3x3 strides=2
t = layers.Conv2D(filters=512,kernel_size=(3,3),strides=2)(t)
t = layers.BatchNormalization(axis=3)(t)
t = layers.Activation('relu')(t)
t = layers.ZeroPadding2D(padding=2)(t)
t = layers.MaxPooling2D(pool_size=2,strides=1)(t)
#4층 3갈래
fourth_1 = layers.Conv2D(filters=512,kernel_size=(4,4),strides=1,padding='SAME')(t)
fourth_1 = layers.BatchNormalization(axis=3)(fourth_1)

fourth_2 = layers.Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='SAME')(t)
fourth_2 = layers.BatchNormalization(axis=3)(fourth_2)

fourth_3 = layers.Conv2D(filters=512,kernel_size=(2,2),strides=1,padding='SAME')(t)
fourth_3 = layers.BatchNormalization(axis=3)(fourth_3)
#4층 Add
t = layers.Add()([fourth_1,fourth_2,fourth_3,t])
t = layers.Activation('relu')(t)
t = layers.Dropout(0.3)(t)
#Last
t = layers.Conv2D(filters=1024,kernel_size=(2,2),strides=1)(t)
t = layers.BatchNormalization(axis=3)(t)
t = layers.Activation('relu')(t)
t = layers.MaxPooling2D(pool_size=2,strides=2)(t)
#Dense
t = layers.GlobalAveragePooling2D()(t)
t = layers.Dense(units=1024,activation='relu')(t)
t = layers.Dense(units=512,activation='relu')(t)
t = layers.Dense(units=2,activation='softmax')(t)

model = tf.keras.models.Model(inputs=[input],outputs=t)
model.summary()
