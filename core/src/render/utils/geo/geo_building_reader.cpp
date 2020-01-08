#include "render/utils/geo/geo_building_reader.h"
#include "render/engine/common/error.h"

#include "zcommon/config/megawise_config.h"


namespace zilliz {
namespace render {
namespace engine {

void GeoBuildingReader::Open() {
    dataset_ = (GDALDataset *) GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (dataset_ == nullptr) {
        RENDER_ENGINE_LOG_ERROR << "Open building file failed";
        return;
    }
}


void GeoBuildingReader::Read() {

    if (dataset_ == nullptr) {
        RENDER_ENGINE_LOG_ERROR << "Dataset is empty.";
        return;
    }

    layer_ = dataset_->GetLayer(0);

    OGRFeature *feature;

    layer_->ResetReading();
    while ((feature = layer_->GetNextFeature()) != nullptr) {
        auto geometry = feature->GetGeometryRef();
        if (geometry != nullptr && wkbFlatten(geometry->getGeometryType()) == wkbPolygon) {
            auto polygon = geometry->toPolygon()->getExteriorRing();
            buildings_.addGeometry(polygon);
        } else if (geometry != nullptr && wkbFlatten(geometry->getGeometryType()) == wkbMultiPolygon) {
            auto multiPolygon = geometry->toMultiPolygon();
            for (auto polygon : multiPolygon) {
                buildings_.addGeometry(polygon);
            }
        } else {
            printf("illegal geometry type\n");
        }
        OGRFeature::DestroyFeature(feature);
    }
}


const std::string GeoBuildingReader::ReadJsonById(const int64_t &geo_id, GeoType geo_type) {

    layer_ = dataset_->GetLayer(0);

    OGRFeature *feature;

    std::string field_name;
    switch (geo_type) {
        case GeoType::kBuilding : field_name = "bin";
            break;
        case GeoType::kBlock : field_name = "geoid10";
            break;
        case GeoType::kDistrict : field_name = "tractid";
            break;
        default: {
            return nullptr;
        }
    }

    layer_->ResetReading();
    while ((feature = layer_->GetNextFeature()) != nullptr) {
        auto id = feature->GetFieldAsInteger64(field_name.c_str());
        if (id == geo_id) {
            auto geometry = feature->GetGeometryRef();
            return geometry->exportToJson();
        }
        OGRFeature::DestroyFeature(feature);
    }

    RENDER_ENGINE_LOG_ERROR << "No geometry found by geo id = " << geo_id;
    return nullptr;
}


//const std::string GeoBuildingReader::GetFilePathByGeoType(const GeoType &geo_type) {
//    switch (geo_type) {
//        case GeoType::kBuilding : {
//            auto building_file_path = utils::megawise::DevCfg::render_engine::building_file_path();
//            return std::string(building_file_path);
//        }
//        case GeoType::kBlock : {
//            auto block_file_path = utils::megawise::DevCfg::render_engine::block_file_path();
//            return std::string(block_file_path);
//        }
//        case GeoType::kDistrict : {
//            auto district_file_path = utils::megawise::DevCfg::render_engine::block_file_path();
//            return std::string(district_file_path);
//        }
//        default: {
//            RENDER_ENGINE_LOG_ERROR << "Unknown geo type:" << geo_type;
//            return nullptr;
//        }
//    }
//}


void GeoBuildingReader::Translate(BoundingBox bound_box) {

    auto poDS = GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == nullptr) {
        RENDER_ENGINE_LOG_ERROR << "Open building file failed";
        return;
    }

    char **papszArgv = nullptr;

    // Clip input layer with a bounding box.
    // argv: -spat <xmin> <ymin> <xmax> <ymax>
    papszArgv = CSLAddString(papszArgv, "-spat");
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box.longitude_left).c_str());
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box.latitude_left).c_str());
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box.longitude_right).c_str());
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box.latitude_right).c_str());

    auto psOptions = GDALVectorTranslateOptionsNew(papszArgv, nullptr);
    std::string pszDest = "/tmp/tmp.shp";
    dataset_ = (GDALDataset *) GDALVectorTranslate(pszDest.c_str(), nullptr, 1, &poDS, psOptions, nullptr);

    if (dataset_ == nullptr) {
        RENDER_ENGINE_LOG_ERROR << "Translate failed.";
        return;
    }

    GDALVectorTranslateOptionsFree(psOptions);
    GDALClose(poDS);
}


void GeoBuildingReader::Close() {
    GDALClose(dataset_);
}


} // namespace engine
} // namespace render
} // namespace zilliz