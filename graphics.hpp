/**
 * @file graphics.hpp
 * @author Marcus Edel
 *
 * Miscellaneous graphics routines.
 */
#include <string>

#include <mlpack/core.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace graphics {
  /**
   * Transform armadillo dense matrix into a jpeg image.
   *
   * @param m - The armadillo dense matrix.
   * @param quality - Jpeg image quality from 0 to 100 (the higher is the
   *                  better). Default value is 95.
   * @return String representation of the jpeg image.
   */
  static inline std::string Mat2Image(const arma::mat& m,
                                      const int quality = 55)
  {
    arma::mat image = m;
    cv::Mat opencvMat(image.n_rows, image.n_cols, CV_64FC1, image.memptr());

    std::vector<int> p {CV_IMWRITE_JPEG_QUALITY, quality};

    std::vector<unsigned char> buf;
    imencode(".jpg", opencvMat, buf, p);

    return std::string(buf.begin(), buf.end());
  }
}