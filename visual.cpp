#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

int main() {
    ifstream f("MiniLLM_v9.cb", ios::binary);
    if (!f) {
        cerr << "Cannot open file\n";
        return 1;
    }

    while (!f.eof()) {
        int rows, cols;
        if (!f.read((char*)&rows, sizeof(int))) break;
        if (!f.read((char*)&cols, sizeof(int))) break;

        vector<float> vals(rows * cols);
        f.read((char*)vals.data(), vals.size() * sizeof(float));

        cout << "Tensor (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << vals[i * cols + j] << " ";
            }
            cout << "\n";
        }
        cout << "----------------------------\n";
    }
    return 0;
}
