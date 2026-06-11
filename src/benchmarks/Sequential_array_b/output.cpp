#include "market.hpp"

// Output (time,price,fundamental_value,log_return,fundamental_shock,log_mispricing,sentiment,optimists,pessimists,fundamentalists,excess_demand)
void write_header(std::ofstream& out) {
    out << "time,"
        << "price,"
        << "fundamental_value,"
        << "log_return,"
        << "fundamental_shock,"
        << "log_mispricing,"
        << "sentiment,"
        << "optimists,"
        << "pessimists,"
        << "fundamentalists"
        << "\n";
}

void write_row(
    std::ofstream& out,
    int t,
    Market market,
    double new_price,
    double epsilon,
    Counts counts
) {
    double log_return = std::log(new_price / market.p);
    double log_mispricing = std::log(market.pf / market.p);
    int noise_traders = counts.nb_optimists + counts.nb_pessimists;
    double sentiment = 0.0;
    if (noise_traders > 0) {
        sentiment =
            static_cast<double>(counts.nb_optimists - counts.nb_pessimists)
            / static_cast<double>(noise_traders);
    }

    out << t << ","
        << new_price << ","
        << market.pf << ","
        << log_return << ","
        << epsilon << ","
        << log_mispricing << ","
        << sentiment << ","
        << counts.nb_optimists << ","
        << counts.nb_pessimists << ","
        << counts.nb_fundamentalists
        << "\n";
}