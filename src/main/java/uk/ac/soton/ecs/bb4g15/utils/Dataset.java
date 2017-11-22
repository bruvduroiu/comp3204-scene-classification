package uk.ac.soton.ecs.bb4g15.utils;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

import java.io.File;

import static org.openimaj.image.ImageUtilities.FIMAGE_READER;

public class Dataset {
    private static final File TRAINING_FOLDER = new File("data/training");
    private static final File TESTING_FOLDER = new File("data/testing");

    public static VFSGroupDataset<FImage> loadTraining() throws FileSystemException {
        return new VFSGroupDataset<FImage>(TRAINING_FOLDER.getAbsolutePath(), FIMAGE_READER);
    }

    public static VFSListDataset<FImage> loadTesting() throws FileSystemException {
        return new VFSListDataset<>(TESTING_FOLDER.getAbsolutePath(), FIMAGE_READER);
    }
}
