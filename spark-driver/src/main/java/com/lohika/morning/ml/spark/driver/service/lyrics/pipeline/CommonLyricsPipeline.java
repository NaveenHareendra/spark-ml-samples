package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.types.*;

public abstract class CommonLyricsPipeline implements LyricsPipeline {

    @Autowired
    protected SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    @Value("${product.test.set.csv.file.path}")
    private String lyricsTestSetDirectoryPath;

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Value("${lyrics.model.file.csv}")
    private String fileName;

    @Override
    public GenrePrediction predict(final String unknownLyrics) {
        String lyrics[] = unknownLyrics.split("\\r?\\n");

        Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics),
           Encoders.STRING());

        Dataset<Row> unknownLyricsDataset = lyricsDataset
                .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                .withColumn(ID.getName(), functions.lit("unknown.txt"));

        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());

        getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);
        Row predictionRow = predictionsDataset.first();

        System.out.println("\n------------------------------------------------");
        final Double prediction = predictionRow.getAs("prediction");
        System.out.println("Prediction: " + Double.toString(prediction));

        if (Arrays.asList(predictionsDataset.columns()).contains("probability")) {
            final DenseVector probability = predictionRow.getAs("probability");
            System.out.println("Probability: " + probability);
            System.out.println("------------------------------------------------\n");

            return new GenrePrediction(getGenre(prediction).getName(), probability.apply(0)
                    , probability.apply(2), probability.apply(6), probability.apply(4)
                    , probability.apply(5), probability.apply(3), probability.apply(1));
        }

        System.out.println("------------------------------------------------\n");
        return new GenrePrediction(getGenre(prediction).getName());
    }


    Dataset<Row> readLyrics() {
//        Dataset input = readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.METAL)
//                                                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.POP));
        Dataset input =readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.POP)
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.ROCK))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.BLUES))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.HIPHOP))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.JAZZ))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.REGGAE))
                .union(readLyricsForGenre(lyricsTrainingSetDirectoryPath, Genre.COUNTRY));

        // Reduce the input amount of partition minimal amount (spark.default.parallelism OR 2, whatever is less)
        input = input.coalesce(sparkSession.sparkContext().defaultMinPartitions()).cache();

        // Force caching.
        input.count();

        return input;
    }

    private Dataset<Row> readLyricsForGenre(String inputDirectory, Genre genre) {
//        System.out.println("Value is "+Genre.METAL.getName());
//        Dataset<Row> lyrics = readLyrics(inputDirectory, genre.name().toLowerCase() + "/*");

        Dataset<Row> lyrics = readLyrics(inputDirectory, fileName, genre );

        Dataset<Row> labeledLyrics = lyrics.withColumn(LABEL.getName(), functions.lit(genre.getValue()));//        lyrics=splitData(lyrics, genre, 42, 0.8, 0.2);

        if(genre.getName()!="hip hop"){
            labeledLyrics=splitData(labeledLyrics, genre, 42, 0.78, 0.22);
        }
        labeledLyrics=labeledLyrics.limit(3700);

//        labeledLyrics.where("genre='rock'").show();
//        labeledLyrics.show();
//        System.out.println(genre.name() + " music sentences = " + lyrics.count());

        return labeledLyrics;
    }

    public Double testAccuracyUnknown(){
        Dataset<Row> rawLyrics = sparkSession.read()
                .option("header", "true")
                .csv(lyricsTestSetDirectoryPath+ "/**/*.csv");
        rawLyrics.show();

        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());
        rawLyrics=rawLyrics.withColumnRenamed("lyrics", VALUE.getName());
        rawLyrics = rawLyrics
                .withColumnRenamed("lyrics", VALUE.getName())
                .withColumn("label", rawLyrics.col("label").cast(DataTypes.DoubleType))
                .select("id", "value", "label");
        rawLyrics.show();
//        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictions = bestModel.transform(rawLyrics);

        predictions.select( "label", "prediction", "probability").show(false);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + accuracy);

        return accuracy;
    }

    Dataset<Row> splitData(Dataset<Row> dataOriginal, Genre genre, long seedVal,
                                   double trainSplitRatio, double testSplitRaio ){

        dataOriginal= dataOriginal.withColumn("rand", functions.rand())  // add a random column
                .orderBy(functions.rand())
                .drop("rand");               // drop it after shuffling;

        //Starting the split
        //Save testing dataset
        String testDataFile;
        if(genre!=null){
            testDataFile = lyricsTrainingSetDirectoryPath+"/test_data/"+genre.getName();
        }else{
            testDataFile = lyricsTrainingSetDirectoryPath+"/test_data/";
        }


        double[] weights = {0.8, 0.2};//80percent : 20 percent split

        long seed = seedVal;


        Dataset<Row>[] splits = dataOriginal.randomSplit(weights, seed);

        Dataset<Row> trainRawLyrics = splits[0];

        Dataset<Row> testRawLyrics= splits[1];

        testRawLyrics.write()
                .format("csv")
                .option("header", "true")
                .mode(SaveMode.Overwrite)
                .csv(testDataFile);

        System.out.println("Test dataFrame saved to CSV directory: " + testDataFile);
        System.out.println("Train Set: " + String.valueOf(trainRawLyrics.count()));
        System.out.println("Test Set: " + String.valueOf(testRawLyrics.count()));

        return trainRawLyrics;
//        split ends
    }

    private Dataset<Row> readLyrics(String inputDirectory, String file, Genre genre) {
//        System.out.println("SPARK SESSION RUNNING DIRECTORY IS: "+inputDirectory);

        Dataset<Row> rawLyrics = sparkSession.read()
                .option("header", "true")
                .csv(inputDirectory+file);

        rawLyrics = rawLyrics.filter(rawLyrics.col("lyrics").notEqual(""));
        rawLyrics = rawLyrics.filter(rawLyrics.col("lyrics").contains(" "));
        rawLyrics=rawLyrics.withColumnRenamed("lyrics", VALUE.getName());

        rawLyrics = rawLyrics.where(rawLyrics.col("genre").equalTo(genre.getName()));

        // Add source filename column as a unique id.
        Dataset<Row> lyrics = rawLyrics.withColumn(ID.getName(), functions.input_file_name());
//        lyrics=lyrics.limit(1000);
        return lyrics;
    }

    private Genre getGenre(Double value) {
        for (Genre genre: Genre.values()){
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }

        return Genre.UNKNOWN;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Arrays.sort(model.avgMetrics());
        modelStatistics.put("Best model metrics", model.avgMetrics()[model.avgMetrics().length - 1]);

        return modelStatistics;
    }

    void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

    void saveModel(CrossValidatorModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    void saveModel(PipelineModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    public void setLyricsTrainingSetDirectoryPath(String lyricsTrainingSetDirectoryPath) {
        this.lyricsTrainingSetDirectoryPath = lyricsTrainingSetDirectoryPath;
    }

    public void setLyricsModelDirectoryPath(String lyricsModelDirectoryPath) {
        this.lyricsModelDirectoryPath = lyricsModelDirectoryPath;
    }

    protected abstract String getModelDirectory();

    String getLyricsModelDirectoryPath() {
        return lyricsModelDirectoryPath;
    }
}
