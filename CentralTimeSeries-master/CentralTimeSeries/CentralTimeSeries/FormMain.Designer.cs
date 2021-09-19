namespace CentralTimeSeries
{
    partial class FormMain
    {
        /// <summary>
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// </summary>
        /// <param name="disposing"> </param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.lbxData = new System.Windows.Forms.ListBox();
            this.btnClear = new System.Windows.Forms.Button();
            this.cmbDataName = new System.Windows.Forms.ComboBox();
            this.btnDPDP = new System.Windows.Forms.Button();
            this.btnDPDP_cla = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // lbxData
            // 
            this.lbxData.FormattingEnabled = true;
            this.lbxData.ItemHeight = 12;
            this.lbxData.Location = new System.Drawing.Point(12, 12);
            this.lbxData.Name = "lbxData";
            this.lbxData.Size = new System.Drawing.Size(652, 460);
            this.lbxData.TabIndex = 1;
            // 
            // btnClear
            // 
            this.btnClear.Location = new System.Drawing.Point(589, 571);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(75, 23);
            this.btnClear.TabIndex = 2;
            this.btnClear.Text = "Clear";
            this.btnClear.UseVisualStyleBackColor = true;
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            // 
            // cmbDataName
            // 
            this.cmbDataName.FormattingEnabled = true;
            this.cmbDataName.Items.AddRange(new object[] {
            "50words",
            "Adiac",
            "Beef",
            "CBF",
            "Coffee",
            "ECG200",
            "FaceAll",
            "FaceFour",
            "fish",
            "Gun_Point",
            "Lighting2",
            "Lighting7",
            "OliveOil",
            "OSULeaf",
            "SwedishLeaf",
            "synthetic_control",
            "Trace",
            "Two_Patterns",
            "wafer",
            "yoga"});
            this.cmbDataName.Location = new System.Drawing.Point(127, 501);
            this.cmbDataName.Name = "cmbDataName";
            this.cmbDataName.Size = new System.Drawing.Size(121, 20);
            this.cmbDataName.TabIndex = 4;
            this.cmbDataName.SelectedIndexChanged += new System.EventHandler(this.cmbDataName_SelectedIndexChanged);
            // 
            // btnDPDP
            // 
            this.btnDPDP.Location = new System.Drawing.Point(475, 537);
            this.btnDPDP.Name = "btnDPDP";
            this.btnDPDP.Size = new System.Drawing.Size(75, 23);
            this.btnDPDP.TabIndex = 6;
            this.btnDPDP.Text = "gDPDP";
            this.btnDPDP.UseVisualStyleBackColor = true;
            this.btnDPDP.Click += new System.EventHandler(this.btn_gDPDP_Click);
            // 
            // btnDPDP_cla
            // 
            this.btnDPDP_cla.Location = new System.Drawing.Point(475, 571);
            this.btnDPDP_cla.Name = "btnDPDP_cla";
            this.btnDPDP_cla.Size = new System.Drawing.Size(75, 23);
            this.btnDPDP_cla.TabIndex = 10;
            this.btnDPDP_cla.Text = "mDPDP";
            this.btnDPDP_cla.UseVisualStyleBackColor = true;
            this.btnDPDP_cla.Click += new System.EventHandler(this.btn_mDPDP_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(14, 542);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(461, 12);
            this.label1.TabIndex = 11;
            this.label1.Text = "Central time series of two samples (initially x=[1,2,3,4,5],y=[3,4,5,6,7]): ";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(14, 576);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(395, 12);
            this.label2.TabIndex = 12;
            this.label2.Text = "Central time series of multiple samples of the selected dataset :";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(14, 504);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(107, 12);
            this.label3.TabIndex = 13;
            this.label3.Text = "Select a dataset:";
            // 
            // FormMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(685, 608);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.btnDPDP_cla);
            this.Controls.Add(this.btnDPDP);
            this.Controls.Add(this.cmbDataName);
            this.Controls.Add(this.btnClear);
            this.Controls.Add(this.lbxData);
            this.Name = "FormMain";
            this.Text = "Central Time Series";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListBox lbxData;
        private System.Windows.Forms.Button btnClear;
        private System.Windows.Forms.ComboBox cmbDataName;
        private System.Windows.Forms.Button btnDPDP;
        private System.Windows.Forms.Button btnDPDP_cla;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
    }
}

