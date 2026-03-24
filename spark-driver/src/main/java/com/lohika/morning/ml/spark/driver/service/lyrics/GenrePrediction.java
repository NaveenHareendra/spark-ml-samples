package com.lohika.morning.ml.spark.driver.service.lyrics;

import org.apache.spark.sql.execution.columnar.DOUBLE;

public class GenrePrediction {

    private String genre;
    private Double rockProbability;
    private Double popProbability;
    private Double countryProbability;
    private Double hipHopProbability;
    private Double jazzProbability;
    private Double reggaeProbability;
    private Double blueProbability;

    public GenrePrediction(String genre, Double rockProbability
            , Double countryProbability, Double hipHopProbability,
            Double jazzProbability, Double reggaeProbability,
            Double blueProbability, Double popProbability) {
        this.genre = genre;
        this.rockProbability = rockProbability;
        this.popProbability = popProbability;
        this.countryProbability = countryProbability;
        this.hipHopProbability = hipHopProbability;
        this.jazzProbability = jazzProbability;
        this.reggaeProbability = reggaeProbability;
        this.blueProbability = blueProbability;
    }

    public GenrePrediction(String genre) {
        this.genre = genre;
    }

    public String getGenre() {
        return genre;
    }

    public Double getBlueProbability() {
        return blueProbability;
    }

    public Double getReggaeProbability() {
        return reggaeProbability;
    }

    public Double getjJazzProbability() {
        return jazzProbability;
    }

    public Double getCountryProbability() {
        return countryProbability;
    }

    public Double getHipHopProbability() {
        return hipHopProbability;
    }

    public Double getRockProbability() {
        return rockProbability;
    }

    public Double getPopProbability() {
        return popProbability;
    }

}
