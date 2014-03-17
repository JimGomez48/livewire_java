import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_highgui;

import java.util.*;

/**
 * WRITE CLASS DESCRIPTION HERE
 */
public class CostMap
{
    public class Node{
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
    }

    private static final float RAD2 = 1.41421356f;
    Node[][] original;
    Node[][] costs;

    public CostMap(CvMat image){
        reset(image);
    }

    public void reset(CvMat image){
        original = new Node[image.rows()][image.cols()];

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
    }

    private enum ExpandMethod{COST, DIST}

    public void addSeed(int row, int col){
        ExpandMethod method = ExpandMethod.COST;

        //TODO keep track of seed points

        reset();
        switch (method){
            case COST: expandVcost(row, col); break;
            case DIST: expandVdist(row, col); break;
            default: expandVcost(row, col);
        }
    }

    private void expandVdist(int row, int col){
        //TODO
    }

    private void expandVcost(int row, int col){
        final String PROG_TITLE = "Expanding Graph...";
        CvMat image = CvMat.create(
                original.length, original[0].length, opencv_core.CV_8UC1, 1);
        opencv_core.cvSetZero(image);
        LivewireApp.showImage(PROG_TITLE, image);

        //create algorithm data structures COMPARE VIA CUM COST
        Set<Node> closed = new HashSet<Node>();
        PriorityQueue<Node> wavefront = new PriorityQueue<Node>(1000, new Comparator<Node>() {
            @Override
            public int compare(Node a, Node b)
            {
                if (a.cost > b.cost) return 1;
                if (a.cost < b.cost) return -1;
                return 0;
            }
        });

        //get seed point, initialize cost to 0, and add to wavefront
        Node current = costs[row][col];
        current.cost = 0;
        wavefront.add(current);

        int count = 0;
        int step = (original.length * original[0].length)/50;
        System.out.println("Expanding graph...");
        while (!wavefront.isEmpty()){
            //get next lowest cost Node from wavefront and add to closed set
            current = wavefront.poll();
            closed.add(current);
            image.put(current.row, current.col, 50);

            //get neighbors of current and expandVcost
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
                    image.put(n.row, n.col, 255);
                }
            }

            if (count % step == 0){
                opencv_highgui.cvShowImage(PROG_TITLE, image);
                opencv_highgui.cvWaitKey(1);
            }
            count++;
        }
        System.out.println("Done");
        opencv_highgui.cvShowImage(PROG_TITLE, image);
        opencv_highgui.cvWaitKey(1);
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

    private int euclideanDist(Node a, Node b){
        return (int)Math.sqrt( ((b.row - a.row) * (b.row - a.row)) +
                ((b.col - a.col) * (b.col- a.col)) );
    }

}
