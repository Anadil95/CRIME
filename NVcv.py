from pyspark.sql import SparkSession
from pyspark.ml.feature import (VectorAssembler,OneHotEncoder,StringIndexer)
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('MyFirstStandaloneApp').setMaster('local')
sc = SparkContext(conf=conf)




spark = SparkSession.builder.appName('EventCode').getOrCreate()
train = spark.read.csv('Crime-12.csv', inferSchema=True,header=True)



Actor1_indexer =  StringIndexer(inputCol='Actor1Code',outputCol='Actor1Code_indexer').setHandleInvalid("keep")
Actor1Code_encoder = OneHotEncoder(inputCol='Actor1Code_indexer',outputCol='Actor1CodeVec')

Actor2Code_indexer =  StringIndexer(inputCol='Actor2Code',outputCol='Actor2Code_indexer').setHandleInvalid("keep")
Actor2Code_encoder = OneHotEncoder(inputCol='Actor2Code_indexer',outputCol='Actor2CodeVec')

Actor1Country_indexer =  StringIndexer(inputCol='Actor1Geo_CountryCode',outputCol='Actor1Country_indexer').setHandleInvalid("keep")
Actor1Country_encoder = OneHotEncoder(inputCol='Actor1Country_indexer',outputCol='Actor1CountryVec')

Actor2Country_indexer =  StringIndexer(inputCol='Actor2Geo_CountryCode',outputCol='Actor2Country_indexer').setHandleInvalid("keep")
Actor2Country_encoder = OneHotEncoder(inputCol='Actor2Country_indexer',outputCol='Actor2CountryVec')

Region_indexer =  StringIndexer(inputCol='Region',outputCol='Region_indexer').setHandleInvalid("keep")
Region_encoder = OneHotEncoder(inputCol='Region_indexer',outputCol='RegionVec')

EventCode_indexer =  StringIndexer(inputCol='EventCode',outputCol='EventCode_indexer').setHandleInvalid("keep")
EventCode_encoder = OneHotEncoder(inputCol='EventCode_indexer',outputCol='EventCodeVec')

assembler = VectorAssembler(inputCols=['EventBaseCode', 'EventRootCode','QuadClass','Actor1CodeVec','Actor1CountryVec','Actor1Geo_Type','Actor2CodeVec'
    ,'RegionVec','Actor2CountryVec','Month'],outputCol='features')


(trainingData, testData) = train.randomSplit([0.7, 0.3])

NV  = NaiveBayes(labelCol="EventCode_indexer", featuresCol="features",modelType="multinomial")


paramGrid = ParamGridBuilder()\
  .build()


evaluator =  MulticlassClassificationEvaluator(labelCol='EventCode_indexer',
                                               predictionCol='prediction',metricName='accuracy')



pipeline = Pipeline(stages=[Actor1_indexer, Actor2Code_indexer, Actor1Country_indexer, Actor2Country_indexer,
                            Region_indexer, Actor1Code_encoder, Actor2Code_encoder, Actor1Country_encoder,
                            Actor2Country_encoder,EventCode_indexer, Region_encoder, assembler, NV])


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=10)  # use 3+ folds in practice

cvModel = crossval.fit(trainingData)

prediction = cvModel.transform(testData)
pp_Acc = evaluator.evaluate(prediction)

result = prediction.select("EventCode_indexer", "prediction", "probability")
result.show()
print('A Naive Bayes algorithm had an accuracy of: {0:2.2f}%'.format(pp_Acc*100))
