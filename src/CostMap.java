import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_highgui;

import java.io.Serializable;
import java.util.*;

/**
 * WRITE CLASS DESCRIPTION HERE
 */
public class CostMap
{
    public class Node implements Cloneable{
        public short row;
        public short col;
        public int cost;
        public Node parent;

        public Node(){}

        public Node(short row, short col, int cost, Node parent){
            this.row = row;
            this.col = col;
            this.cost = cost;
            this.parent = parent;
        }

        @Override
        protected Object clone() throws CloneNotSupportedException
        {
            return new Node(this.row, this.col, this.cost, this.parent);
        }
    }

    private static final float RAD2 = 1.41421356f;
    Node[][] original;
    Node[][] costs;

    public CostMap(CvMat image){
        reset(image);
    }

    public void reset(CvMat image){
        original = new Node[image.rows()][image.cols()];
//        costs = new Node[image.rows()][image.cols()];

        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                Node n = new Node();
                n.row = (short)i;
                n.col = (short)j;
                n.cost = (int)image.get(i, j);
                n.parent = null;
                original[i][j] = n;
            }
        }

//        reset();
    }

    public void reset(){
        costs = Arrays.copyOf(original, original.length);

//        for (int i = 0; i < original.length; i++) {
//            for (int j = 0; j < original[0].length; j++) {
//                try {
//                    costs[i][j] = (Node)original[i][j].clone();
//                }
//                catch (CloneNotSupportedException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
    }

    public void addSeed(int row, int col){
        //TODO
        reset();
        expand(row, col);
    }

    private void expand(int row, int col){
        final String PROG_TITLE = "Expanding Graph...";
        CvMat expandImage = CvMat.create(
                original.length, original[0].length, opencv_core.CV_8UC1, 1);
        opencv_core.cvSetZero(expandImage);
        LivewireApp.showImage(PROG_TITLE, expandImage);

        //create algorithm data structures
        Set<Node> closed = new HashSet<Node>();
        PriorityQueue<Node> wavefront = new PriorityQueue<Node>(1000, new NodeCompare());

        //get seed point, initialize cost to 0, and add to wavefront
        Node current = costs[row][col];
        current.cost = 0;
        wavefront.add(current);

        int count = 0;
        int step = (original.length * original[0].length)/100;
        System.out.println("Expanding graph...");
        while (!wavefront.isEmpty()){
            //get next lowest cost Node from wavefront and add to closed set
            current = wavefront.poll();
            closed.add(current);
            expandImage.put(current.row, current.col, 50);

            //get neighbors of current and expand
            ArrayList<Node> neigbors = getNeighbors(current);
            for (int i = 0; i < neigbors.size(); i++) {
                Node n = neigbors.get(i);
                if (closed.contains(n)) continue;

                int tentativeCost = euclideanAdd(current, n);
                if (n.parent == null || current.cost + tentativeCost < n.cost){
                    n.parent = current;
                    n.cost = tentativeCost;
                }

                //add neighbors to wavefront if not already in
                if (!wavefront.contains(n)){
                    wavefront.add(n);
                    expandImage.put(n.row, n.col, 255);
                }
            }

            if (count % step == 0){
                opencv_highgui.cvShowImage(PROG_TITLE, expandImage);
                opencv_highgui.cvWaitKey(1);
            }
            count++;
        }
        System.out.println("Done");
        opencv_highgui.cvShowImage(PROG_TITLE, expandImage);
        opencv_highgui.cvWaitKey(1);
    }

    private class NodeCompare implements Comparator<Node>, Serializable{
        @Override
        public int compare(Node a, Node b)
        {
            if (a.cost > b.cost) return 1;
            if (a.cost < b.cost) return -1;
            return 0;
        }
    }

    private ArrayList<Node> getNeighbors(Node n){
        ArrayList<Node>neighbors = new ArrayList<Node>(8);

        for (int i = n.row-1; i <= n.row+1; i++){
            for (int j = n.col-1; j <= n.col+1; j++){
                //skip node n
                if (i == n.row && j == n.col) continue;
                //check if current neighbor is within image bounds
                if (i >= 0 && j >= 0 && i < costs.length && j < costs[0].length){
                    neighbors.add(costs[i][j]);
                }
            }
        }

        return neighbors;
    }

    private int euclideanAdd(Node current, Node n){
        //if diagonal, scale by RAD2
        if (current.row != n.row && current.col != n.col)
            return current.cost + (int)(RAD2 * costs[n.row][n.col].cost);

        return current.cost + costs[n.row][n.col].cost;
    }

//    private int toIndex(int row, int col){
//        return original[0].length * row + col;
//    }
//
//    private int toIndex(Node n){
//        return original[0].length * n.row + n.col;
//    }
}
