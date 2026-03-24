package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {

    ROCK("rock", 0D),

    POP("pop", 1D),

    COUNTRY("country", 2D),

    BLUES("blues", 3D),

    JAZZ("jazz", 4D),

    REGGAE("reggae", 5D),

    HIPHOP("hip hop", 6D),

    UNKNOWN("Dont know :(", -1D);

    private final String name;
    private final Double value;

    Genre(final String name, final Double value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Double getValue() {
        return value;
    }

}
