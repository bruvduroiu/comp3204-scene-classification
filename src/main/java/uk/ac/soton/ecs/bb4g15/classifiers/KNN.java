package uk.ac.soton.ecs.bb4g15.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

import uk.ac.soton.ecs.bb4g15.App;
import uk.ac.soton.ecs.bb4g15.utils.MListDataset;

public class KNN extends Classifier {
    private int k;

    public KNN(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing, int k) {
    	super(training, testing);
        this.k = k;
    }

    private static double euclideanDistance(double[] instance1, double[] instance2) {
        return Math.sqrt(
                IntStream.range(0, instance1.length)
                        .mapToDouble(i -> Math.pow(instance1[i] - instance2[i],2))
                        .sum());
    }

    //Crops the image to a square and resizes it to 16x16
    private static FImage cropAndResize(FImage image) {
    	int cropDimensions = Math.min(image.getWidth(), image.getHeight());
    	image = crop(image, cropDimensions, cropDimensions);
        image = image.processInplace(new ResizeProcessor(16, 16));
        return image;
    }

    //Gets the class with the shortest distance
    private static String getVote(ArrayList<Map.Entry<String, Double>> distances, int k) {
        distances.sort(Comparator.comparing(Map.Entry::getValue));
        Map<String, Integer> votes = new HashMap<>();
        for (Map.Entry<String, Double> neighbour : distances.subList(0, k))
            votes.put(neighbour.getKey(),
                    (votes.containsKey(neighbour.getKey())) ? votes.get(neighbour.getKey()) + 1 : 1);

        List<Map.Entry<String,Integer>> votesList = new LinkedList<>(votes.entrySet());
        Collections.sort(votesList, Comparator.comparing(Map.Entry::getValue));
        return votesList.get(0).getKey();
    }

    //Trains the model
    public Map<String, String> _train(Map<String, String> classifications) {
        ArrayList<Map.Entry<String, DoubleFV>> trainingFeatures = new ArrayList<>();
        ArrayList<Map.Entry<String, DoubleFV>> testingFeatures = new ArrayList<>();

        App.log("Calculating training features");
        for (Map.Entry<String, MListDataset<FImage>> entry : training.entrySet()) {
            MListDataset<FImage> images = entry.getValue();

            for (int i = 0; i < images.size(); i++) {
                FImage resized = cropAndResize(images.getInstance(i));
                //adds the centered, normalised vector of the reduced image
                trainingFeatures.add(new HashMap.SimpleEntry<>(entry.getKey(), centerAndNormaliseVector(resized.getDoublePixelVector())));
            }
        }

        App.log("Calculating testing features");
        for (int i = 0; i < testing.size(); i++) {
            FImage resized = cropAndResize(testing.getInstance(i));
            testingFeatures.add(new HashMap.SimpleEntry<>(testing.getID(i), centerAndNormaliseVector(resized.getDoublePixelVector())));
        }

        App.log("Testing KNN");
        for (Map.Entry<String, DoubleFV> testVectEntry : testingFeatures) {
            ArrayList<Map.Entry<String, Double>> distances = new ArrayList<>();
            String testId = testVectEntry.getKey();
            DoubleFV testVec = testVectEntry.getValue();

            //finds the distance from the test vector to the other vectors
            for (Map.Entry<String, DoubleFV> trainFV : trainingFeatures)
                distances.add(new HashMap.SimpleEntry<>(trainFV.getKey(), euclideanDistance(testVec.asDoubleVector(), trainFV.getValue().asDoubleVector())));

            String prediction = getVote(distances, k);

            classifications.put(testId, prediction);
        }
        return classifications;
    }
}
