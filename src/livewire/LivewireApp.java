package livewire;

import com.googlecode.javacpp.Pointer;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_highgui;
import com.googlecode.javacv.cpp.opencv_imgproc;

/**
 * <p>An implementation of the "Live-Wire" image segmentation tool, also known as
 * "Intelligent Scissors".</p>
 *
 * <p>Based on the paper "Interactive live-wire boundary
 * extraction" by William A. Barrett and Eric N. Mortensen.</p>
 *
 * @author James Gomez
 */
public class LivewireApp
{
    private static final String APP_TITLE = "Live-Wire";
    private CvMat origImage;
    private CvMat image;
    private CostMap costMap;

    private class GradStruct
    {
        public CvMat x;
        public CvMat y;
        public CvMat mag;
        public CvMat dir;
    }

    public LivewireApp(String path) {
        try {
            IplImage temp = opencv_highgui.cvLoadImageBGRA(path);
            origImage = temp.asCvMat();
            image = opencv_highgui.cvLoadImageM(path, opencv_core.CV_8U);
            System.out.println("Loaded image \"" + path + "\" type:" +
                    typeToString(origImage.type()));
        }
        catch (Exception e) {
            System.out.println("ERROR: could not load file " + path);
//            e.printStackTrace();
            System.exit(1);
        }
    }

    public void run() {
        GradStruct gradient = getGradient(image);
        CvMat edges = getEdges(image);
        CvMat sum = getWeightedSum(gradient, edges, 0.80f, 0.25f, 0.15f);
        costMap = new CostMap(sum);
//        showFeatures(gradient, edges, sum);

        showImage(APP_TITLE, origImage, 100, 100);
        opencv_highgui.cvSetMouseCallback(APP_TITLE, new MouseCallback(), null);
        opencv_highgui.cvWaitKey(0);
    }

    private GradStruct getGradient(CvMat image) {
        //Istantiate Gradient Mats
        GradStruct gradient = new GradStruct();
        gradient.x = CvMat.create(image.rows(), image.cols(), image.type());
        gradient.y = CvMat.create(image.rows(), image.cols(), image.type());
        gradient.mag = CvMat.create(image.rows(), image.cols(), image.type());
        gradient.dir = CvMat.create(image.rows(), image.cols(), image.type());

        //Copy image and convert to 16-bit signed data
        CvMat temp = CvMat.create(image.rows(), image.cols(), opencv_core.CV_16S, 1);
        temp.put(image);

        //Blur image to reduce noise
        opencv_imgproc.GaussianBlur(
                temp, temp,
                new opencv_core.CvSize(3, 3),
                0, 0,
                opencv_imgproc.BORDER_DEFAULT
        );

        CvMat gx = CvMat.create(image.rows(), image.cols(), opencv_core.CV_16S, 1);
        CvMat gy = CvMat.create(image.rows(), image.cols(), opencv_core.CV_16S, 1);
        int ddepth = opencv_core.CV_16SC1;
        int scale = 1;
        int delta = 0;

        /*//LAPLACIAN
        opencv_imgproc.Laplacian(temp, gradient.mag, -1, 5, 1, 0,
        opencv_imgproc.BORDER_DEFAULT);
        opencv_core.cvNot(gradient.mag, gradient.mag);*/

        //SOBEL get Gx and Gy
        opencv_imgproc.Sobel(temp, gx, ddepth, 1, 0, 3,
                scale, delta, opencv_imgproc.BORDER_DEFAULT);
        opencv_imgproc.Sobel(temp, gy, ddepth, 0, 1, 3,
                scale, delta, opencv_imgproc.BORDER_DEFAULT);

        //Scale and shift Gx and Gy values for viewing, and get Gradient Magnitude
        opencv_core.cvConvertScale(gx, gradient.x, 1.0 / 2.0, 128);
        opencv_core.cvConvertScale(gy, gradient.y, 1.0 / 2.0, 128);
        CvMat gxMag = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U, 1);
        CvMat gyMag = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U, 1);
        opencv_core.cvConvertScaleAbs(gx, gxMag, 1, 0);
        opencv_core.cvConvertScaleAbs(gy, gyMag, 1, 0);
        opencv_core.cvAddWeighted(gxMag, 1.0, gyMag, 1.0, 0, gradient.mag);

        //Get Gradient Direction
        CvMat dir = CvMat.create(image.rows(), image.cols(), opencv_core.CV_16U, 1);
        for (int i = 0; i < image.rows(); i++) {
            for (int j = 0; j < image.cols(); j++) {
                float angle = (float) Math.toDegrees(Math.atan2(gy.get(i, j),
                        gx.get(i, j)));
                dir.put(i, j, angle);
            }
        }
        gradient.dir.put(dir);

        return gradient;
    }

    private CvMat getEdges(CvMat image) {
        CvMat edges = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U, 1);
        edges.put(image);
        opencv_imgproc.GaussianBlur(
                edges, edges,
                new opencv_core.CvSize(3, 3),
                0, 0,
                opencv_imgproc.BORDER_DEFAULT
        );
        opencv_imgproc.Canny(edges, edges, 15, 45, 3, true);

        return edges;
    }

    private CvMat getWeightedSum(
            GradStruct grad, CvMat edges, float wg, float wz, float wd)
    {
        CvMat sum = CvMat.create(image.rows(), image.cols(), image.type(), 1);
        CvMat gradMag = CvMat.create(image.rows(), image.cols(),
                opencv_core.CV_8U, 1);
        CvMat graddir = CvMat.create(image.rows(), image.cols(),
                opencv_core.CV_8U, 1);
        CvMat gEdges = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U,
                1);
        opencv_core.cvScale(grad.mag, gradMag, wg, 0.0);
        opencv_core.cvScale(grad.dir, graddir, wd, 0.0);
        opencv_core.cvScale(edges, gEdges, wz, 0.0);

        opencv_core.cvAdd(gradMag, graddir, sum, null);
        opencv_core.cvAdd(sum, gEdges, sum, null);
        opencv_core.cvNot(sum, sum);

        return sum;
    }

    public static void showImage(String title, CvMat image) {
        opencv_highgui.cvNamedWindow(title);
        opencv_highgui.cvShowImage(title, image);
    }

    public static void showImage(String title, CvMat image, int x, int y) {
        opencv_highgui.cvNamedWindow(title);
        opencv_highgui.cvMoveWindow(title, x, y);
        opencv_highgui.cvShowImage(title, image);
    }

    private static final String GRAD_X_TITLE = "Gradient X-component";
    private static final String GRAD_Y_TITLE = "Gradient Y-component";
    private static final String GRAD_MAG_TITLE = "Gradient Magnitude";
    private static final String GRAD_DIR_TITLE = "Gradient Direction";
    private static final String EDGES_TITLE = "Canny Edges";
    private static final String COST_MAP_TITLE = "Cost map (1 - weighted sum)";

    private void showFeatures(GradStruct gradient, CvMat edges, CvMat sum) {
        if (gradient != null) {
            if (gradient.x != null)
                showImage(GRAD_X_TITLE, gradient.x, 600, 100);
            if (gradient.y != null)
                showImage(GRAD_Y_TITLE, gradient.y, 1100, 100);
            if (gradient.mag != null)
                showImage(GRAD_MAG_TITLE, gradient.mag, 100, 600);
            if (gradient.dir != null)
                showImage(GRAD_DIR_TITLE, gradient.dir, 600, 600);
        }

        if (edges != null)
            showImage(EDGES_TITLE, edges, 1100, 600);
        if (sum != null)
            showImage(COST_MAP_TITLE, sum, 800, 300);
    }

    private void printMatType(String name, CvMat m) {
        System.out.println(name + "->type: (" + m.type() + ") " +
                typeToString(m.type()));
    }

    private String typeToString(int type) {
        String r;
        int depth = type & opencv_core.CV_MAT_DEPTH_MASK;
        int chans = 1 + (type >> opencv_core.CV_CN_SHIFT);

        switch (depth) {
            case opencv_core.CV_8U: r = "8U"; break;
            case opencv_core.CV_8S: r = "8S"; break;
            case opencv_core.CV_16U: r = "16U"; break;
            case opencv_core.CV_16S: r = "16S"; break;
            case opencv_core.CV_32S: r = "32S"; break;
            case opencv_core.CV_32F: r = "32F"; break;
            case opencv_core.CV_64F: r = "64F"; break;
            default: r = "User"; break;
        }

        return r + "C" + chans;
    }

    private class MouseCallback extends opencv_highgui.CvMouseCallback
    {
        private boolean seedset;

        private CvPoint nextPoint;
        private CvPoint currentPoint;
        private CostMap.Node seedNode;
        private CvMat lines;
        private CvMat liveImage;

        public MouseCallback() {
            seedset = false;
            nextPoint = new CvPoint();
            currentPoint = new CvPoint();

            lines = CvMat.create(origImage.rows(), origImage.cols(),
                    origImage.type(), origImage.channels());
            liveImage = CvMat.create(origImage.rows(), origImage.cols(),
                    origImage.type(), origImage.channels());

            opencv_core.cvZero(lines);
            opencv_core.cvZero(liveImage);
        }

        @Override
        public void call(int event, int x, int y, int flags,
                         Pointer param) {
            switch (event) {
                case opencv_highgui.CV_EVENT_LBUTTONDOWN:
                    seedNode = costMap.getNode(y, x);
                    costMap.addSeed(y, x);
                    nextPoint.put(x, y);
                    seedset = true;
                    break;
                case opencv_highgui.CV_EVENT_LBUTTONUP:
                    break;
                case opencv_highgui.CV_EVENT_LBUTTONDBLCLK:
                    break;
                case opencv_highgui.CV_EVENT_RBUTTONDOWN:
                    seedset = false;
                    break;
                case opencv_highgui.CV_EVENT_RBUTTONUP:
                    break;
                case opencv_highgui.CV_EVENT_RBUTTONDBLCLK:
                    System.out.println("Cleared current boundary");
                    opencv_core.cvZero(lines);
                    break;
                case opencv_highgui.CV_EVENT_MOUSEMOVE:
                    if (seedset) {
                        opencv_core.cvZero(lines);
                        CostMap.Node current = costMap.getNode(y, x);
                        currentPoint.put(current.col, current.row);
                        nextPoint.put(current.parent.col, current.parent.row);
                        while (!current.equals(seedNode) || current == null){
                            opencv_core.cvDrawLine(lines, currentPoint, nextPoint,
                                    CvScalar.YELLOW, 1, 8, 0);
                            current = current.parent;
                            currentPoint.put(current.col, current.row);
                            if (current.parent != null)
                                nextPoint.put(current.parent.col, current.parent.row);
                        }
                    }
                    break;
            }
            opencv_core.cvAdd(origImage, lines, liveImage, null);
            opencv_highgui.cvShowImage(APP_TITLE, liveImage);
            opencv_highgui.cvWaitKey(1);

            //super.call(event, x, y, flags, param);
        }
    }

    private static final String USAGE = "USAGE: <executable> <path to image file>";

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("No image data\n" + USAGE);
            return;
        }

        LivewireApp app = new LivewireApp(args[0]);
        app.run();
    }

}
