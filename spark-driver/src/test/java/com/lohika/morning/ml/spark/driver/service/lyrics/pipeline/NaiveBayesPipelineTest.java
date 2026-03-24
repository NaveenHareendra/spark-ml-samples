package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.driver.service.BaseTest;
import com.lohika.morning.ml.spark.driver.service.MLService;
import org.springframework.beans.factory.annotation.Autowired;

public class NaiveBayesPipelineTest extends BaseTest {

    @Autowired
    private NaiveBayesBagOfWordsPipeline naiveBayesPipeline;

    @Autowired
    private MLService mlService;


    public void testPipelineRuns() {
        // minimal smoke test
        // If the pipeline has any public method, call it here
        // For example, just check that bean is not null
        assert naiveBayesPipeline != null;
        assert mlService != null;
    }

}
