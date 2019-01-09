from keras.models import load_model, Input, Model
import keras

def ensembleModels(models, model_input):
    yModels=[model(model_input) for model in models] 
    yAvg= keras.layers.average(yModels) 
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')   
    return modelEns

def main():
    kdef = '../trained_models/emotion_models/mini_XCEPTION_KDEF.hdf5'
    fer = '../trained_models/emotion_models/fer2013_mini_XCEPTION.107-0.66.hdf5'
    modelPaths = [kdef, fer]
    modelNames = ['kdef', 'fer']

    models=[]
    for mP, mN in zip(modelPaths, modelNames):
        modelTemp=load_model(mP, compile=False)
        modelTemp.name=mN
        models.append(modelTemp)

    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
    modelEns = ensembleModels(models, model_input)
    modelEns.save('../trained_models/emotion_models/ensemble_boost_model.hdf5')
    modelEns.summary()

if __name__ == '__main__':
    main()
