/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gis/gdal/geometry_visitor.h"

namespace zilliz {
namespace gis {
namespace gdal {

void NPointsVisitor::visit(const OGRPoint* geo) {
  if (geo->IsEmpty()) return;
  npoints_++;
}

double PrecisionReduceVisitor::coordinate_precision_reduce(double coordinate){
    std::string coordinate_string = std::to_string(coordinate);
    std::string precision_reduce_coordinate;
    int32_t num_0_to_add = 0;

    if (int64_t(coordinate_string.find(".")) == -1 || int64_t(coordinate_string.find(".")) > precision_) {
      if (int64_t(coordinate_string.find(".")) == -1){
        num_0_to_add = coordinate_string.length() - precision_;
      }
      else{
        num_0_to_add = coordinate_string.find(".") - precision_;
      }
      
      if (coordinate_string.length() <= precision_) {
        precision_reduce_coordinate = coordinate_string;
      }
      else{

        if (int32_t(coordinate_string[precision_] - 48) < 5) {
          precision_reduce_coordinate = coordinate_string.substr(0, precision_);
          for(int32_t i = 0; i < num_0_to_add; i++){
            precision_reduce_coordinate += "0";
          }
        }
        else{
         double value_of_coordinate = std::stod(coordinate_string.substr(0, precision_)) + 1;
         precision_reduce_coordinate = std::to_string(value_of_coordinate).substr(0, precision_);
         for (int32_t i = 0; i < num_0_to_add; i++) {
           precision_reduce_coordinate += "0";
         }     
        }

      }
    }
    else{
     if (coordinate_string.find(".") == precision_) {
       if (int32_t(coordinate_string[precision_ + 1] - 48) < 5) {
         precision_reduce_coordinate = coordinate_string.substr(0, precision_);
       }
       else{
         double value_of_coordinate = std::stod(coordinate_string.substr(0, precision_));
         precision_reduce_coordinate = std::to_string(value_of_coordinate + 1).substr(0, precision_); 
       }
     }
     else{
       if (coordinate_string.length() <= precision_ + 1){
         precision_reduce_coordinate = coordinate_string;
       }
       else{
         if(int32_t(coordinate_string[precision_ + 1] - 48 ) < 5){
           precision_reduce_coordinate = coordinate_string.substr(0, precision_ + 1);
         }
         else{
           double value_of_coordinate = std::stod(coordinate_string.substr(0, precision_ + 1));
           double carry_value = 1;
           for (int32_t i = 0; i < (precision_ - coordinate_string.find(".")); i++) {
             carry_value /= 10;
           }
           precision_reduce_coordinate = std::to_string(value_of_coordinate + carry_value).substr(0, precision_ + 1);
         }
       }
     }
    }
    
    return std::stod(precision_reduce_coordinate);    
}

void PrecisionReduceVisitor::visit(OGRPoint* geo){
    double coordinate_x = geo->getX();
    double coordinate_y = geo->getY();
    geo->setX(coordinate_precision_reduce(coordinate_x));
    geo->setY(coordinate_precision_reduce(coordinate_y));
}



}  // namespace gdal
}  // namespace gis
}  // namespace zilliz
