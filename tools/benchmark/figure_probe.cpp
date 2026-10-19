// Native test driver; contains no ATAT implementation or source.
#include "FigureObjective.hpp"
#include <iostream>
#include <iomanip>
int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    if (argc == 1) {
        int n;
        while (std::cin >> n) {
            FigureObjective f; f.mode = "atat";
            std::cin >> f.tolerance >> f.wr >> f.wn >> f.wd;
            std::vector<double> values(n);
            for (int i=0;i<n;++i) {
                FigureObjective::Component c;
                std::cin >> c.order >> c.diameter >> c.target >> values[i];
                f.components.push_back(c);
            }
            std::cout << f.score(values) << '\n';
        }
    } else {
        FigureObjective f; f.load(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
        std::vector<int> occ(f.sites);
        while (std::cin >> occ[0]) {
            for (int i=1;i<f.sites;++i) std::cin >> occ[i];
            auto values=f.correlations(occ);
            for (double v:values) std::cout << v << ' ';
            double worst=0;
            int tested=0, limit=argc>4 ? std::stoi(argv[4]) : f.sites*f.sites;
            for (int a=0;a<f.sites && tested<limit;++a) for(int b=a+1;b<f.sites && tested<limit;++b) {
                ++tested;
                auto proposed=f.swapped(occ,values,a,b);
                std::swap(occ[a],occ[b]); auto full=f.correlations(occ);
                std::swap(occ[a],occ[b]);
                for (size_t i=0;i<full.size();++i) worst=std::max(worst,std::abs(full[i]-proposed[i]));
            }
            std::cout << worst << '\n';
        }
    }
}
