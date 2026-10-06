// Headless Directional raw face-field generator for strand preprocessing.
//
// This program uses the official Directional library to compute a face-based
// 2-RoSy power field on an OBJ mesh, converts it to a raw extrinsic field, and
// writes the transposed DMAT matrix expected by the downstream data loader.

#include <directional/CartesianField.h>
#include <directional/PCFaceTangentBundle.h>
#include <directional/TriMesh.h>
#include <directional/power_field.h>
#include <directional/power_to_raw.h>
#include <directional/principal_matching.h>
#include <directional/readOBJ.h>

#include <Eigen/Core>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_dmat(const std::string& path, const Eigen::MatrixXd& matrix) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open output DMAT: " + path);
    }
    out << matrix.cols() << "\n";
    out << matrix.rows() << "\n";
    out << std::scientific << std::setprecision(17);
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            out << matrix(row, col);
            if (col + 1 < matrix.cols()) {
                out << " ";
            }
        }
        out << "\n";
    }
}

Eigen::MatrixXd initial_constraint_vector(const directional::TriMesh& mesh) {
    Eigen::MatrixXd const_vectors(1, 3);
    Eigen::RowVector3d edge = mesh.V.row(mesh.F(0, 2)) - mesh.V.row(mesh.F(0, 1));
    const double norm = edge.norm();
    if (norm <= 1e-12) {
        edge = mesh.V.row(mesh.F(0, 1)) - mesh.V.row(mesh.F(0, 0));
    }
    if (edge.norm() <= 1e-12) {
        edge << 1.0, 0.0, 0.0;
    }
    const_vectors.row(0) = edge.normalized();
    return const_vectors;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <input.obj> <rawFaceField.dmat> [degree]\n";
        return 2;
    }
    const std::string mesh_path = argv[1];
    const std::string output_path = argv[2];
    const int degree = argc == 4 ? std::stoi(argv[3]) : 2;
    if (degree != 2) {
        std::cerr << "Only degree=2 is supported for strand tangent preprocessing.\n";
        return 2;
    }

    directional::TriMesh mesh;
    if (!directional::readOBJ(mesh_path, mesh)) {
        std::cerr << "Failed to read OBJ: " << mesh_path << "\n";
        return 1;
    }
    if (mesh.F.rows() == 0) {
        std::cerr << "Mesh has no faces: " << mesh_path << "\n";
        return 1;
    }

    directional::PCFaceTangentBundle face_bundle;
    face_bundle.init(mesh);

    Eigen::VectorXi const_faces(1);
    const_faces << 0;
    Eigen::MatrixXd const_vectors = initial_constraint_vector(mesh);
    Eigen::VectorXd align_weights = Eigen::VectorXd::Constant(1, -1.0);

    directional::CartesianField power_face_field;
    directional::CartesianField raw_face_field;
    directional::power_field(face_bundle, const_faces, const_vectors, align_weights, degree, power_face_field, true);
    directional::power_to_raw(power_face_field, degree, raw_face_field, true);
    directional::principal_matching(raw_face_field);

    if (raw_face_field.extField.rows() != mesh.F.rows() || raw_face_field.extField.cols() != 3 * degree) {
        std::cerr << "Unexpected raw field shape: " << raw_face_field.extField.rows() << " x "
                  << raw_face_field.extField.cols() << ", expected " << mesh.F.rows() << " x " << 3 * degree << "\n";
        return 1;
    }

    // The data loader transposes this matrix and reshapes it to
    // (num_faces, 2, 3), so write extField.T with shape (6, num_faces).
    write_dmat(output_path, raw_face_field.extField.transpose());
    std::cout << "wrote " << output_path << " with raw Directional field shape "
              << raw_face_field.extField.rows() << " x " << raw_face_field.extField.cols() << "\n";
    return 0;
}
