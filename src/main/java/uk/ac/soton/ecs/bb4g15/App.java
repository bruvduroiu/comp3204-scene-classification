package uk.ac.soton.ecs.bb4g15;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

import java.awt.*;
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.openimaj.image.ImageUtilities.FIMAGE_READER;

public class App {

    private static double euclideanDistance(double[] instance1, double[] instance2) {
        return Math.sqrt(
                IntStream.range(0, instance1.length)
                        .mapToDouble(i -> Math.pow(instance1[i] - instance2[i],2))
                        .sum());
    }

    private static FImage cropAndResize(FImage image) {
        int cropDimensions = Math.min(image.getWidth(), image.getHeight());
        Point center = new Point(image.getWidth()/2, image.getHeight()/2);

        float[][] crop = new float[cropDimensions][cropDimensions];

        int xStart = (center.x - cropDimensions/2);
        int yStart = (center.y - cropDimensions/2);

        for (int y = yStart; y < (center.y + cropDimensions/2); y++) {
            for (int x = xStart; x < (center.x + cropDimensions/2); x++) {
                int cropX = x - xStart;
                int cropY = y - yStart;
                crop[cropY][cropX] = image.pixels[y][x];
            }
        }
        image = image.internalAssign(new FImage(crop));
        image = image.processInplace(new ResizeProcessor(16, 16));
        return image;
    }

    private static DoubleFV normaliseVector(double[] vector) {
        double mean = DoubleStream.of(vector).average().getAsDouble();
        double length = Math.sqrt(DoubleStream.of(vector)
                .map(i -> Math.pow(i, 2))
                .sum());

        vector = DoubleStream.of(vector)
                .map(i -> (i / mean) / length)
                .toArray();
        return new DoubleFV(vector);
    }

    private static void run1() {
        try {

            // Resize for getting square images
            VFSGroupDataset<FImage> training =
                    new VFSGroupDataset<>("/Users/bogdanbuduroiu/Development/comp_vision/cw3/scene-classification/data/training",
                            FIMAGE_READER);
            VFSListDataset<FImage> testing =
                    new VFSListDataset<>("/Users/bogdanbuduroiu/Development/comp_vision/cw3/scene-classification/data/testing",
                            FIMAGE_READER);

            ArrayList<Map.Entry<String, DoubleFV>> trainingFeatures = new ArrayList<>();
            ArrayList<Map.Entry<String, DoubleFV>> testingFeatures = new ArrayList<>();
            for (Map.Entry<String, VFSListDataset<FImage>> entry : training.entrySet()) {
                VFSListDataset<FImage> images = entry.getValue();

                for (int i = 0; i < images.size(); i++) {
                    FImage resized = cropAndResize(images.getInstance(i));
                    trainingFeatures.add(new HashMap.SimpleEntry<>(entry.getKey(), normaliseVector(resized.getDoublePixelVector())));
                }
            }

            for (int i = 0; i < testing.size(); i++) {
                FImage resized = cropAndResize(testing.getInstance(i));
                testingFeatures.add(new HashMap.SimpleEntry<>(testing.getID(i), normaliseVector(resized.getDoublePixelVector())));
            }

            // Train KNN
            for (Map.Entry<String, DoubleFV> testVectEntry : testingFeatures) {
                ArrayList<Map.Entry<String, Double>> distances = new ArrayList<>();
                String testId = testVectEntry.getKey();
                DoubleFV testVec = testVectEntry.getValue();

                for (Map.Entry<String, DoubleFV> trainFV : trainingFeatures)
                    distances.add(new HashMap.SimpleEntry<>(trainFV.getKey(), euclideanDistance(testVec.asDoubleVector(), trainFV.getValue().asDoubleVector())));

                distances.sort(Comparator.comparing(Map.Entry::getValue));
                System.out.println(testId + ": " + distances.get(0).getKey());
            }

        } catch (FileSystemException e) {
            e.printStackTrace();
        }
    }

    public static void main( String[] args ) {
        run1();
    }
}
