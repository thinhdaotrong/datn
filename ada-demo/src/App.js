import { useState } from "react";
import "./App.css";
import { Row, Col, Form, Input, Button, Select, Upload } from "antd";
import "antd/dist/antd.css";
import axios from "axios";

const { Option } = Select;

function App() {
  const [form] = Form.useForm();

  const [result0, setResult0] = useState(null);
  console.log("🚀 ~ file: App.js ~ line 13 ~ App ~ result0", result0);
  const [result1, setResult1] = useState(null);
  console.log("🚀 ~ file: App.js ~ line 13 ~ App ~ result1", result1);
  const [result2, setResult2] = useState(null);
  console.log("🚀 ~ file: App.js ~ line 15 ~ App ~ result2", result2);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const onChangeFile = ({ file, fileList }) => {
    console.log(file);
    setFile(file);
  };

  const onFinish = (values) => {
    const { so_vong_lap, tham_so_svm, ti_le_test } = values;
    let formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    setResult0(null);
    setResult1(null);
    setResult2(null);
    axios({
      method: "post",
      url: `http://localhost:1702/query1?m=${so_vong_lap}&c=${tham_so_svm}&instance_categorization=False&percent_test=${ti_le_test}`,
      data: formData,
    }).then((res) => {
      setResult0(res.data.data);
    });
    axios({
      method: "post",
      url: `http://localhost:1702/query?m=${so_vong_lap}&c=${tham_so_svm}&instance_categorization=False&percent_test=${ti_le_test}`,
      data: formData,
    }).then((res) => {
      setResult1(res.data.data);
    });
    axios({
      method: "post",
      url: `http://localhost:1702/query?m=${so_vong_lap}&c=${tham_so_svm}&instance_categorization=True&percent_test=${ti_le_test}`,
      data: formData,
    }).then((res) => {
      setResult2(res.data.data);
      setLoading(false);
    });
  };
  return (
    <div className="container">
      <h1>Weighted SWM + Adaboost</h1>
      <div className="">
        <h2>Đầu vào</h2>
        <Form
          form={form}
          layout="vertical"
          onFinish={onFinish}
          initialValues={{
            so_vong_lap: 50,
            tham_so_svm: 100,
            ti_le_test: 0.3,
          }}
        >
          <Form.Item label="Bộ dữ liệu" name="file">
            <Upload
              showUploadList={false}
              onChange={onChangeFile}
              multiple={false}
              beforeUpload={() => false}
            >
              <Button style={{ marginRight: 20 }}>Upload</Button>
              {file && <span>{file.name}</span>}
            </Upload>
          </Form.Item>
          <Form.Item label="Số vòng lặp Adaboost" name="so_vong_lap">
            <Input />
          </Form.Item>
          <Form.Item label="Tham số C trong SVM" name="tham_so_svm">
            <Input />
          </Form.Item>
          <Form.Item label="Tỉ lệ tập kiểm thử" name="ti_le_test">
            <Input />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              Submit
            </Button>
          </Form.Item>
        </Form>
      </div>
      <div>
        <h2>Kết quả</h2>
        <Row>
          {result0 && (
            <Col md={8}>
              <h3>SVM</h3>
              <div className="result">
                <span>Precision:</span>
                <span>{result0["precision"]["1"]}</span>
              </div>
              <div className="result">
                <span>Recall:</span>
                <span>{result0["recall"]["1"]}</span>
              </div>
              <div className="result">
                <span>F1 score:</span>
                <span>{result0["f1-score"]["1"]}</span>
              </div>
            </Col>
          )}
          {result1 && (
            <Col md={8}>
              <h3>SVM + Adaboost</h3>
              <div className="result">
                <span>Precision:</span>
                <span>{result1["precision"]["1"]}</span>
              </div>
              <div className="result">
                <span>Recall:</span>
                <span>{result1["recall"]["1"]}</span>
              </div>
              <div className="result">
                <span>F1 score:</span>
                <span>{result1["f1-score"]["1"]}</span>
              </div>
            </Col>
          )}
          {result2 && (
            <Col md={8}>
              <h3>SVM + Adaboost có thay đổi trọng số</h3>
              <div className="result">
                <span>Precision:</span>
                <span>{result2["precision"]["1"]}</span>
              </div>
              <div className="result">
                <span>Recall:</span>
                <span>{result2["recall"]["1"]}</span>
              </div>
              <div className="result">
                <span>F1 score:</span>
                <span>{result2["f1-score"]["1"]}</span>
              </div>
            </Col>
          )}
        </Row>
      </div>
    </div>
  );
}

export default App;
