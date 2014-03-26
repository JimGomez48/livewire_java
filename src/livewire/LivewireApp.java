package livewire;

import com.googlecode.javacpp.Pointer;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_highgui;
import com.googlecode.javacv.cpp.opencv_imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * <p>An implementation of the "Live-Wire" image segmentation tool, also known as
 * "Intelligent Scissors".</p>
 *
 * <p>Based on the paper "Interactive live-wire coolwire
 * extraction" by William A. Barrett and Eric N. Mortensen.</p>
 *
 * @author James Gomez
 */
public class LivewireApp
{
    private static final String APP_TITLE = "Live-Wire";
    /** Stores an unaltered copy of the original user specified image */
    private CvMat origImage;
    /** A grayscale copy of the origImage for manipulation and feature extraction */
    private CvMat image;
    /** Used for generating and storing least cost boundaries within the image */
    private CostMap costMap;

    /**
     * A structure used to store the gradient x and y components, gradient magnitude,
     * and gradient direction features of an image.
     */
    private class GradStruct
    {
        /** The x-component of the image gradient */
        public CvMat x;
        /** The y-component of the image gradient */
        public CvMat y;
        /** The gradient magnitude of the image */
        public CvMat mag;
        /** The gradient direction of the image */
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
            System.exit(1);
        }
    }

    /**
     * Initializes application classes and structures and creates the application
     * GUI
     */
    public void run() {
        GradStruct gradient = getGradient(image);
        CvMat edges = getEdges(image);
        CvMat sum = getInverseWeightedSum(gradient, edges, 0.80f, 0.25f, 0.15f);
        costMap = new CostMap(sum);
//        showFeatures(gradient, edges, sum);

        showImage(APP_TITLE, origImage, 100, 100);
        opencv_highgui.cvSetMouseCallback(APP_TITLE, new MouseCallback(), null);
        opencv_highgui.cvWaitKey(0);
    }

    /**
     * Exracts the gradient x and y components, gradient magnitude, and gradient
     * direction features from the specified image.
     *
     * @return a structure containing the gradient x and y components, gradient
     * magnitude, and gradient direction
     */
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

    /**
     * Uses the OpenCv Canny Edge detector to extractor the edge features from the
     * image
     *
     * @return the edge features of the image
     */
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

    /**
     * Calculates the inverse of the weighted sum of the gradient and edge features.
     *
     * @param grad  a structure containing the gradient x and y components, the
     *              gradient magnitude, and the gradient direction features
     * @param edges the edge features
     * @param wg    the weight to apply to the gradient magnitude feature
     * @param wz    the weight to apply to the edge feature
     * @param wd    the weight to apply to the gradient direction feature
     * @return the inverse-weighted-sum of the features
     */
    private CvMat getInverseWeightedSum(
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

    /**
     * Creates a named window at an unspecified location and shows the CvMat as an
     * image within that named window
     */
    public static void showImage(String title, CvMat image) {
        opencv_highgui.cvNamedWindow(title);
        opencv_highgui.cvShowImage(title, image);
    }

    /**
     * Creates a named window at the specified point and shows the CvMat as an image
     * within that named window
     */
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

    /**
     * Shows the Gradient, edge, and weighted sum features extracted from the
     * original image, if available.
     */
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

    /** Prints the Mat pixel type of the given CvMat to standard out */
    private void printMatType(String name, CvMat m) {
        System.out.println(name + "->type: (" + m.type() + ") " +
                typeToString(m.type()));
    }

    /** Converts the numerical type code to a human-readable string */
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

    /**
     * An implementation of the CvMouseCallback for capturing mouse events within the
     * Livewire application
     */
    private class MouseCallback extends opencv_highgui.CvMouseCallback
    {
        private boolean seedset;
        private CvPoint currentPoint;
        private CvPoint nextPoint;
        private CostMap.Node seedNode;
        private CvMat livewire;
        private CvMat coolwire;
        private List<CostMap.Node> boundary;

        public MouseCallback() {
            seedset = false;

            currentPoint = new CvPoint();
            nextPoint = new CvPoint();

            livewire = CvMat.create(origImage.rows(), origImage.cols(),
                    origImage.type(), origImage.channels());
            coolwire = CvMat.create(origImage.rows(), origImage.cols(),
                    origImage.type(), origImage.channels());

            livewire.put(origImage);
            coolwire.put(origImage);

            boundary = new ArrayList<CostMap.Node>(2000);
        }

        /**
         * This method is called when mouse events are triggered on the corresponding
         * GUI
         *
         * @param event specifies the type of mouse event that occured
         * @param x     the x position of the mouse when this event was captured
         * @param y     the y position of the mouse when this event was captured
         * @param flags ?
         * @param param ?
         */
        @Override
        public void call(int event, int x, int y, int flags, Pointer param) {
            switch (event) {
                case opencv_highgui.CV_EVENT_LBUTTONDOWN:
                    if (!seedset || seedNode == null) {
                        seedNode = costMap.snapToEdge(y, x, 7);
                        costMap.addSeed(seedNode.row, seedNode.col);
                        seedset = true;
                    }
                    else{
                        CostMap.Node current = costMap.getClosestEdge(y, x);
                        if (coolBoundary(current, seedNode)){
                            System.out.println("Boundary is closed");
                            seedset = false;
                            drawCoolWire();
                            showSegment();
                        }
                        else{
                            drawCoolWire();
                            seedNode = current;
                            costMap.addSeed(seedNode.row, seedNode.col);
                        }
                    }
                    break;
                case opencv_highgui.CV_EVENT_RBUTTONDOWN:
                    seedset = false;
                    break;
                case opencv_highgui.CV_EVENT_RBUTTONDBLCLK:
                    seedset = false;
                    System.out.println("Cleared current boundary");
                    livewire.put(origImage);
                    coolwire.put(origImage);
                    boundary.clear();
                    break;
                case opencv_highgui.CV_EVENT_MOUSEMOVE:
                    if (seedset) {
//                        drawLiveWire(costMap.getNode(y, x), seedNode);
                    }
                    break;
            }
            if (seedset) drawLiveWire(costMap.getNode(y, x), seedNode);
            opencv_highgui.cvShowImage(APP_TITLE, livewire);
            opencv_highgui.cvWaitKey(1);
        }

        /**
         * Cools a portion of the boundary. That portion will no longer be subject to
         * change as the mouse cursor moves
         *
         * @param current the start Node to cool the boundary from
         * @param lastseed the last seedpoint to cool the boundary to.
         * @return true if the cooled boundary is closed, false otherwise
         */
        private boolean coolBoundary(CostMap.Node current, CostMap.Node lastseed){
            boolean closed = false;
            CostMap.Node n = current;
            int redundantCount = 0;
            ArrayList<CostMap.Node> buffer = new ArrayList<CostMap.Node>();
            while (!n.equals(lastseed) && n.parent != null){
                if (!boundary.isEmpty() && boundary.get(0).equals(n))
                    closed = true;

                if (!closed)
                    redundantCount++;

                buffer.add(n);
                n = n.parent;
            }
            Collections.reverse(buffer);
            boundary.addAll(buffer);

            //remove redundant nodes
            if (closed)
                boundary = boundary.subList(0, boundary.size() - redundantCount - 1);

            return closed;
        }

        /**
         * Draws the livewire boundary onto the livewire image
         *
         * @param start the start Node to draw the boundary from
         * @param end   the end Node to draw the boundary to
         */
        private void drawLiveWire(CostMap.Node start, CostMap.Node end) {
            livewire.put(coolwire);
            while (!start.equals(end) && start.parent != null) {
                currentPoint.put(start.col, start.row);
                nextPoint.put(start.parent.col, start.parent.row);
                opencv_core.cvDrawLine(livewire, currentPoint, nextPoint,
                        CvScalar.RED, 1, 8, 0);
                start = start.parent;
            }
        }

        private void drawCoolWire() {
            CostMap.Node n;
            for (int i = 0; i < boundary.size() -1; i++) {
                n = boundary.get(i);
                currentPoint.put(n.col, n.row);
                n = boundary.get(i + 1);
                nextPoint.put(n.col, n.row);
                opencv_core.cvDrawLine(coolwire, currentPoint, nextPoint,
                        CvScalar.YELLOW, 1, 8, 0);
            }
            livewire.put(coolwire);
        }

        private boolean isBoundaryClosed(){
            boolean closed = false;
            CostMap.Node seed = boundary.get(0);
            int count = 0;

            for (int i = boundary.size() - 1; i >= 1; i--) {
                CostMap.Node n = boundary.get(i);
                if (n.equals(seed)) {
                    closed = true;
                }
                //only check the end 20% of the boundary
                if (count >= 0.2*boundary.size()) break;
            }

            return closed;
        }

        private void showSegment(){
//            CvSeq contour = new CvSeq(boundary.size());
//            opencv_core.CvSeqBlock block;
//            CvChain contour = new CvChain(boundary.size());
//            opencv_ml.CvVectors t = new opencv_ml.CvVectors(boundary.size());
            CvMat contour = CvMat.create(origImage.rows(), origImage.cols(), opencv_core.CV_8U);
            CvMat mask = CvMat.create(origImage.rows(), origImage.cols(), opencv_core.CV_8U);
            opencv_core.cvZero(contour);
            opencv_core.cvZero(mask);


            Iterator<CostMap.Node> iter = boundary.iterator();
            while (iter.hasNext()){
                CostMap.Node n = iter.next();
                contour.put(n.row, n.col, 255);
            }
            showImage("Boundary", contour, 800, 100);

//            CvPoint2D32f point = new CvPoint2D32f();
//            for (int i = 0; i < contour.rows(); i++) {
//                for (int j = 0; j < contour.cols(); j++) {
//                    point.put(j, i);
//                    if (opencv_imgproc.cvPointPolygonTest(contour, point, 1) >= 0)
//                        mask.put(i, j, 255);
//                }
//            }
            showImage("Segment Mask", mask, 100, 600);
//            opencv_highgui.cvWaitKey(1);
        }

    }

    private static final String USAGE = "USAGE: <executable> <path to image file>";

    /** The application's entry point */
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("No image data\n" + USAGE);
            return;
        }

        LivewireApp app = new LivewireApp(args[0]);
        printInstructions();
        app.run();
    }

    private static void printInstructions(){
        System.out.println("\nINSTRUCTIONS");
        System.out.println("======================");
        System.out.println("- Click near an edge to generate a seed point");
        System.out.println("- Drag the cursor around to adjust boundary free end");
        System.out.println("- To cool current boundary, left-click near a desired edge");
        System.out.println("- To clear current boundary, double-click right mouse button");
        System.out.println("- To close off boundary, overlap free end with current boundary tail.");
        System.out.println("  The app will detect boundary closure and stop tracing.");
        System.out.println("- Press enter to display the current boundary, or double-click");
        System.out.println("  the right mouse button to clear the current boundary");
        System.out.println();
    }

}
