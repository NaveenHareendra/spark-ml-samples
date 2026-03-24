package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.functions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component("LogisticRegressionPipeline")
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

    @Value("${training.validator.parallelism}")
    private int trainingValidatorParallelism;

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    public CrossValidatorModel classify() {

        Dataset<Row> sentences = readLyrics();

//        sentences= sentences.withColumn("rand", functions.rand())  // add a random column
//                .orderBy(functions.rand())
//                .drop("rand");               // drop it after shuffling;

        sentences=sentences.select("id","value", "label");
//        Dataset<Row> newSplit = sentences.where(sentences.col("label").notEqual(5));
//        sentences =sentences.where(sentences.col("label").equalTo(5));
//        sentences=splitData(sentences, null, 42, 0.8, 0.2);
//        sentences=sentences.union(newSplit);

        sentences.show();
//        System.out.println("Before Sentences count "+String.valueOf(sentences.count()));
        sentences.groupBy("label").count().show();
//        System.out.println("After Sentences count "+String.valueOf(sentences.count()));

        Cleanser cleanser = new Cleanser();

        Numerator numerator = new Numerator();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol(CLEAN.getName())
                .setOutputCol(WORDS.getName());

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol(WORDS.getName())
                .setOutputCol(FILTERED_WORDS.getName());

        Exploder exploder = new Exploder();

        Stemmer stemmer = new Stemmer();

        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol(VERSE.getName())
                .setOutputCol("features")
                .setMinCount(0);

        LogisticRegression logisticRegression = new LogisticRegression()
                .setFamily("multinomial");

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        cleanser,
                        numerator,
                        tokenizer,
                        stopWordsRemover,
                        exploder,
                        stemmer,
                        uniter,
                        verser,
                        word2Vec,
                        logisticRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{ 48})
                .addGrid(word2Vec.vectorSize(), new int[] { 400})
                .addGrid(logisticRegression.regParam(), new double[] {  0.8D})
                .addGrid(logisticRegression.maxIter(), new int[] { 300})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setParallelism(trainingValidatorParallelism)
                .setNumFolds(10);

        CrossValidatorModel model = crossValidator.fit(sentences);

        saveModel(model, getModelDirectory());
//        testUnknownData(model);
        return model;
    }

    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        modelStatistics.put("Sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[8]).getVectors().count());
        modelStatistics.put("Vector size", ((Word2VecModel) stages[8]).getVectorSize());
        modelStatistics.put("Reg parameter", ((LogisticRegressionModel) stages[9]).getRegParam());
        modelStatistics.put("Max iterations", ((LogisticRegressionModel) stages[9]).getMaxIter());

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/logistic-regression/";
    }

}
