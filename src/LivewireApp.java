import com.googlecode.javacpp.Pointer;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_highgui;
import com.googlecode.javacv.cpp.opencv_imgproc;

/**
 * <p>An implementation of the "Live-Wire" image segmentation tool, also known as
 * "Intelligent Scissors".</p>
 *
 * <p>Based on the paper "Interactive live-wire boundary
 * extraction by William A. Barrett and Eric N. Mortensen.</p>
 *
 * @author James Gomez
 */
public class LivewireApp
{
    private static final String APP_TITLE = "Live-Wire";
    private CvMat image;
    private CostMap costMap;

    private class GradStruct{
        public CvMat x;
        public CvMat y;
        public CvMat mag;
        public CvMat dir;
    }

    public LivewireApp(String path)
    {
        image = opencv_highgui.cvLoadImageM(path, opencv_core.CV_8UC1);
        costMap = new CostMap(image);

        showImage(APP_TITLE, image, 100, 100);
        opencv_highgui.cvSetMouseCallback(APP_TITLE, new opencv_highgui.CvMouseCallback(){
            @Override
            public void call(int event, int x, int y, int flags, Pointer param)
            {
                switch (event){
                    case opencv_highgui.CV_EVENT_LBUTTONDOWN:
                        System.out.println("Left Down");
                        break;
                    case opencv_highgui.CV_EVENT_LBUTTONUP:
                        System.out.println("Left Up");
                        break;
                    case opencv_highgui.CV_EVENT_RBUTTONDOWN:
                        System.out.println("Right Down");
                        break;
                    case opencv_highgui.CV_EVENT_RBUTTONUP:
                        System.out.println("Right Up");
                        break;
                    case opencv_highgui.CV_EVENT_LBUTTONDBLCLK:
                        System.out.println("Double Click");
                        break;
                }
//                super.call(event, x, y, flags, param);
            }
        },null);
    }

    public void run()
    {
        GradStruct gradient = getGradient(image);
        CvMat edges = getEdges(image);
        CvMat sum = getWeightedSum(gradient, edges, 0.80f, 0.25f, 0.15f);
        showFeatures(gradient, edges, sum);

        opencv_highgui.cvWaitKey(0);
    }

    private GradStruct getGradient(CvMat image){
        //Istantiate Gradient Mats
        GradStruct gradient = new GradStruct();
        gradient.x = CvMat.create(image.rows(), image.cols(), image.type());
        gradient.y = CvMat.create(image.rows(), image.cols(), image.type());
        gradient.mag = CvMat.create(image.rows(), image.cols(), image.type());
        gradient.dir = CvMat.create(image.rows(), image.cols(), image.type());

        //Copy image and convert to 16-bit signed data
        CvMat temp = CvMat.create(image.rows(), image.cols(), opencv_core.CV_16S, 1);
        temp.put(image);
        printMatType("temp", temp);
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
        opencv_imgproc.Laplacian(temp, gradient.mag, -1, 5, 1, 0, opencv_imgproc.BORDER_DEFAULT);
        opencv_core.cvNot(gradient.mag, gradient.mag);*/

        //SOBEL get Gx and Gy
        printMatType("gx", gx);
        opencv_imgproc.Sobel(temp, gx, ddepth, 1, 0, 3,
                scale, delta, opencv_imgproc.BORDER_DEFAULT);
        opencv_imgproc.Sobel(temp, gy, ddepth, 0, 1, 3,
                scale, delta, opencv_imgproc.BORDER_DEFAULT);
        printMatType("gx", gx);

        //Scale Gx and Gy for viewing, and get Gradient Magnitude
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
                float angle = (float)Math.toDegrees(Math.atan2(gy.get(i, j), gx.get(i, j)));
//                float angle = (float)Math.atan2(gy.get(i, j), gx.get(i, j));
                dir.put(i, j, angle);
            }
        }
        gradient.dir.put(dir);

        return gradient;
    }

    private CvMat getEdges(CvMat image){
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

    private CvMat getWeightedSum(GradStruct grad, CvMat edges, float wg, float wz, float wd){
        CvMat sum = CvMat.create(image.rows(), image.cols(), image.type(), 1);
        CvMat gradMag = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U, 1);
        CvMat graddir = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U, 1);
        CvMat gEdges = CvMat.create(image.rows(), image.cols(), opencv_core.CV_8U, 1);
        opencv_core.cvScale(grad.mag, gradMag, wg, 0.0);
        opencv_core.cvScale(grad.dir, graddir, wd, 0.0);
        opencv_core.cvScale(edges, gEdges, wz, 0.0);

        opencv_core.cvAdd(gradMag, graddir, sum, null);
        opencv_core.cvAdd(sum, gEdges, sum, null);
        opencv_core.cvNot(sum, sum);

        return sum;
    }

    private void showImage(String title, CvMat image){
        opencv_highgui.cvNamedWindow(title);
        opencv_highgui.cvShowImage(title, image);
    }

    private void showImage(String title, CvMat image, int x, int y){
        opencv_highgui.cvNamedWindow(title);
        opencv_highgui.cvMoveWindow(title, x, y);
        opencv_highgui.cvShowImage(title, image);
    }

    private void showFeatures(GradStruct gradient, CvMat edges, CvMat sum){

        if (gradient != null){
            if (gradient.x != null)
                showImage("Gradient x-component", gradient.x, 600, 100);
            if (gradient.y != null)
                showImage("Gradient y-component", gradient.y, 1100, 100);
            if (gradient.mag != null)
                showImage("Gradient Magnitude", gradient.mag, 100, 600);
            if (gradient.dir != null)
                showImage("Gradient Direction", gradient.dir, 600, 600);
        }

        if (edges != null)
            showImage("Canny Edges", edges, 1100, 600);
        if (sum != null)
            showImage("Inverted Sum (Cost Map)", sum, 800, 300);
    }

    private String typeToString(int type){
        String r;
        int depth = type & opencv_core.CV_MAT_DEPTH_MASK;
        int chans = 1 + (type >> opencv_core.CV_CN_SHIFT);

        switch ( depth ) {
            case opencv_core.CV_8U:  r = "8U"; break;
            case opencv_core.CV_8S:  r = "8S"; break;
            case opencv_core.CV_16U: r = "16U"; break;
            case opencv_core.CV_16S: r = "16S"; break;
            case opencv_core.CV_32S: r = "32S"; break;
            case opencv_core.CV_32F: r = "32F"; break;
            case opencv_core.CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

       return r;
    }

    private void printMatType(String name, CvMat m){
        System.out.println(name + "->type: (" + m.type() + ") "  +
                typeToString(m.type()) + "C" + m.channels());
    }

    private void printMat(String name, CvMat m){
        System.out.println(name + ":");
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                System.out.print(m.get(i, j) + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args)
    {
        LivewireApp app = new LivewireApp(args[0]);
        app.run();
    }
}
