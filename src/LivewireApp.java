import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_highgui;

import javax.swing.*;

/**
 * WRITE CLASS DESCRIPTION HERE
 */
public class LivewireApp
{
    private IplImage image;
    private CanvasFrame canvas;

    public LivewireApp(){
        image = opencv_highgui.cvLoadImage("mri_1.jpg");
        canvas = new CanvasFrame("Live-wire", 1);
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public void run(){
        canvas.showImage(image);
    }

    public static void main(String[] args){
        LivewireApp app = new LivewireApp();
        app.run();
    }
}
