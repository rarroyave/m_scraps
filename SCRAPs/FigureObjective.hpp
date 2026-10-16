#ifndef SCRAPS_FIGURE_OBJECTIVE_HPP
#define SCRAPS_FIGURE_OBJECTIVE_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Each component is an orbit average of decorated periodic figures. A term
// may reference one site repeatedly: different images share its occupation.
class FigureObjective {
public:
    struct Term {
        int component;
        double coefficient;
        std::vector<std::pair<int, int>> factors;
    };
    struct Component { int order; double diameter, target; };
    std::vector<Component> components;
    std::vector<Term> terms;
    std::vector<std::vector<double>> basis;
    std::vector<std::vector<int>> incidence;
    std::vector<double> cell, positions;
    std::vector<std::vector<int>> allowedSpecies;
    std::vector<int> siteGroups;
    std::vector<std::vector<int>> siteCounts;
    std::string mode = "weighted_l1";
    double wd = 0.0, wn = 1.0, wr = 1.0, tolerance = 1e-3;
    int sites = 0, species = 0;

    bool active() const { return !components.empty(); }

    void load(const std::string& path, int expectedSites, int expectedSpecies) {
        std::ifstream in(path);
        auto fail = []() { throw std::runtime_error("FIGURES_FILE: invalid or inconsistent figure data"); };
        std::string magic;
        int version = 0, rows = 0, count = 0;
        if (!(in >> magic >> version >> sites >> species >> rows >> count)
            || magic != "SCRAPS_FIGURES" || (version != 1 && version != 2 && version != 3)
            || sites != expectedSites || species != expectedSpecies
            || rows <= 0 || count <= 0) fail();
        auto real = [&]() {
            double v;
            if (!(in >> v) || !std::isfinite(v)) fail();
            return v;
        };
        cell.resize(9);
        positions.resize(3 * sites);
        for (auto& x : cell) x = real();
        for (auto& x : positions) x = real();
        basis.assign(rows, std::vector<double>(species));
        for (auto& row : basis) for (auto& x : row) x = real();
        if (version >= 2) {
            allowedSpecies.resize(sites);
            for (auto& allowed : allowedSpecies) {
                int n;
                if (!(in >> n) || n < 1 || n > species) fail();
                for (int i = 0; i < n; ++i) {
                    int sp;
                    if (!(in >> sp) || sp < 0 || sp >= species) fail();
                    allowed.push_back(sp);
                }
                std::sort(allowed.begin(), allowed.end());
                if (std::adjacent_find(allowed.begin(), allowed.end()) != allowed.end()) fail();
            }
        }
        if (version == 3) {
            siteGroups.resize(sites);
            siteCounts.assign(sites, std::vector<int>(species));
            for (int i = 0; i < sites; ++i) {
                if (!(in >> siteGroups[i]) || siteGroups[i] < -1) fail();
                for (int& count : siteCounts[i]) {
                    if (!(in >> count) || count < 0 || count > sites) fail();
                }
            }
        }
        incidence.assign(sites, {});
        bool hasPairs = false;
        for (int c = 0; c < count; ++c) {
            Component component;
            int nt;
            if (!(in >> component.order) || component.order < 2 || component.order > 4) fail();
            component.diameter = real();
            component.target = real();
            if (!(in >> nt) || nt <= 0 || component.diameter <= 0.0) fail();
            hasPairs |= component.order == 2;
            components.push_back(component);
            double norm = 0.0;
            for (int j = 0; j < nt; ++j) {
                Term term{c, real(), {}};
                if (term.coefficient <= 0.0) fail();
                norm += term.coefficient;
                std::vector<int> affected;
                for (int p = 0; p < component.order; ++p) {
                    int site, function;
                    if (!(in >> site >> function) || site < 0 || site >= sites
                        || function < 0 || function >= rows) fail();
                    term.factors.emplace_back(site, function);
                    affected.push_back(site);
                }
                std::sort(affected.begin(), affected.end());
                affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
                for (int site : affected) incidence[site].push_back(terms.size());
                terms.push_back(std::move(term));
            }
            if (std::abs(norm - 1.0) > 1e-8) fail();
        }
        std::string extra;
        if ((in >> extra) || !hasPairs) fail();
    }

    double product(const Term& term, const std::vector<int>& occupations,
                   int a = -1, int b = -1) const {
        double result = term.coefficient;
        for (const auto& factor : term.factors) {
            int site = factor.first;
            int source = site == a ? b : (site == b ? a : site);
            result *= basis[factor.second][occupations[source]];
        }
        return result;
    }

    std::vector<double> correlations(const std::vector<int>& occupations) const {
        std::vector<double> result(components.size(), 0.0);
        for (const auto& term : terms) result[term.component] += product(term, occupations);
        return result;
    }

    // Computes a proposal without modifying the accepted state. Shared-site
    // terms are visited once, including figures containing both swap sites.
    std::vector<double> swapped(const std::vector<int>& occupations,
                                const std::vector<double>& current, int a, int b) const {
        auto result = current;
        std::vector<int> affected;
        std::set_union(incidence[a].begin(), incidence[a].end(),
                       incidence[b].begin(), incidence[b].end(),
                       std::back_inserter(affected));
        for (int t : affected) {
            const auto& term = terms[t];
            result[term.component] += product(term, occupations, a, b) - product(term, occupations);
        }
        return result;
    }

    double score(const std::vector<double>& correlations) const {
        double d0 = std::numeric_limits<double>::max();
        int maxOrder = 2;
        for (const auto& c : components) {
            d0 = std::min(d0, c.diameter);
            maxOrder = std::max(maxOrder, c.order);
        }
        std::vector<double> ranges(maxOrder - 1, 0.0), errors(components.size());
        for (size_t i = 0; i < components.size(); ++i) {
            const auto& c = components[i];
            errors[i] = std::abs(correlations[i] - c.target);
            ranges[c.order - 2] = std::max(ranges[c.order - 2], c.diameter);
        }
        for (auto& r : ranges) r += d0;
        for (size_t i = 0; i < components.size(); ++i) {
            const auto& c = components[i];
            if (errors[i] > tolerance) ranges[c.order - 2] = std::min(ranges[c.order - 2], c.diameter);
        }
        for (size_t i = 1; i < ranges.size(); ++i) ranges[i] = std::min(ranges[i], ranges[i - 1]);
        double d1 = *std::min_element(ranges.begin(), ranges.end());
        // Rescale all weights by the same factor to avoid underflow without
        // changing the normalized L1 score. ATAT's near-zero test below uses
        // the unscaled numerator, as in its 3.52 source.
        double maxLogWeight = -std::numeric_limits<double>::infinity();
        for (const auto& c : components) {
            if (mode != "atat" || c.diameter >= d1 - 1e-3)
                maxLogWeight = std::max(maxLogWeight, -wd * c.diameter / d0 + (c.order - 2) * std::log(wn));
        }
        double numerator = 0.0, denominator = 0.0;
        for (size_t i = 0; i < components.size(); ++i) {
            const auto& c = components[i];
            if (mode == "atat" && c.diameter < d1 - 1e-3) continue;
            double weight = std::exp(-wd * c.diameter / d0 + (c.order - 2) * std::log(wn) - maxLogWeight);
            numerator += weight * errors[i];
            denominator += weight;
        }
        if (mode == "atat" && (denominator == 0.0 || numerator == 0.0
            || std::log(numerator) + maxLogWeight < std::log(1e-3)))
            return -std::numeric_limits<float>::max();
        double result = numerator / denominator;
        if (mode == "atat") {
            for (size_t p = 0; p < ranges.size(); ++p) result -= wr * std::pow(wn, p) * ranges[p] / d0;
        }
        return result;
    }
};
#endif
